import transformers
import torch
from typing import Optional, Union


def prepare_masked_batch(
    texts: list[str],
    num_masks: Union[int, float],
    rng: torch.Generator,
    tokenizer: transformers.tokenization_utils_fast.PreTrainedTokenizerFast,
    device: torch.device,
    disallowed_ids: Optional[list[int]] = None,
) -> tuple[torch.LongTensor, torch.Tensor]:
    """_summary_
    Takes a list of strings, tokenizes them, and for each one, masks a random subset of the tokens.

    Args:
        texts (list[str]): A list of texts on which to perform masking.
        num_masks (Union[int,float]): The number of tokens to substitute with masks. If a float p between 0,1 is given, then masking is done with probability p.
        rng (torch.Generator): A random number generator for selecting the tokens to mask.
        tokenizer (transformers.tokenization_utils_fast.PreTrainedTokenizerFast): The text tokenizer.
        device (torch.device): which device to store the tokenized tensors on.
        disallowed_ids (Optional[list[int]], optional): A list of additional tokens to ignore -- special tokens are always ignored. Defaults to None.

    Returns:
        tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]: Returns the token id tensor and attention mask, with shapes [batch size x longest sentence], [batch size x longest sentence], and [batch size x longest sentence x 4].
        Substitutions tensor has the following format:
        (i,j,0) = in sentence i, mask number j, which token in the sentence was masked
        (i,j,1) = what was the original token?
        (i,j,2) = -1, but will be used to track final token choice.
        (i,j,3) = -1 but will be used later to track when this mask token was unmasked.

    """
    # Tokens we are not allowed to convert to <mask>!
    if disallowed_ids is None:
        disallowed_ids = []
    disallowed_ids += (
        tokenizer.all_special_ids
    )  # [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]
    disallowed_ids = torch.tensor(
        disallowed_ids, dtype=torch.int64, device=device
    ).unique()

    # for each row, generate a set of legal tokens to mask.
    # for each row, use rng.choice to choose the appropriate indices to mask.
    # mask those tokens, noting the substitutions made.
    # substitutions should be a [batch_size x num_masks x 3] tensor
    # (i,j,0) = in sentence i, mask # j, which token in the sentence was masked
    # (i,j,1) = what was the original token?
    # (i,j,2) = -1, but will be used to track final token choice.

    mask_id = tokenizer.mask_token_id

    tokenized = tokenizer(texts, padding=True)
    token_tensor = torch.tensor(
        tokenized["input_ids"], dtype=torch.int64, device=device
    )
    attention_tensor = torch.tensor(
        tokenized["attention_mask"], dtype=torch.float32, device=device
    )

    # I will iterate over all of the sentences, because I do not know that the number of allowed tokens in each sentece will be the same
    # This complicates the use of rng.choice(num_allowed_tokens, num_masked, False)
    batch_size = len(texts)

    # now, we need to decide the number of masks for each text.
    num_masks_sent = torch.zeros(batch_size, dtype=torch.int64, device=device)
    if num_masks > 1:
        num_masks = int(num_masks)
    if type(num_masks) == int:
        num_masks_sent[:] = num_masks
    else:
        mask_probability = torch.tensor(num_masks, device=device).float()
        num_masks_sent[:] = torch.sum(
            ~torch.isin(token_tensor[:, :], disallowed_ids), axis=1
        )  # count the number of tokens that are allowed to be masked.
        # print('test: ', torch.binomial(num_masks_sent.float(),mask_probability,generator=rng))
        num_masks_sent[:] = torch.binomial(
            num_masks_sent.float(), mask_probability, generator=rng
        ).long()

    substitutions = torch.zeros(
        (batch_size, torch.max(num_masks_sent), 4), dtype=torch.int64, device=device
    )
    substitutions[:, :, 0] = (
        -1
    )  # to deal with the fact that there may be different numbers of tokens to mask in each sentence, we will substitution rounds with nothing to be -1.
    for sentence_ind in range(batch_size):
        num_masks = num_masks_sent[sentence_ind]
        if num_masks == 0:
            continue
        allowed_tokens_mask = ~torch.isin(token_tensor[sentence_ind, :], disallowed_ids)
        indices = torch.nonzero(allowed_tokens_mask).squeeze()
        # print(indices.shape[0], rng,device)
        subs_inds = torch.arange(num_masks, device=device)
        token_indices_to_mask, _ = torch.sort(
            indices[
                torch.randperm(indices.shape[0], generator=rng, device=device)[
                    subs_inds
                ]
            ]
        )
        # print("num masks: ", num_masks)
        # print("allowed tokens mask: ", allowed_tokens_mask)
        # print("token indices to mask: ", token_indices_to_mask)
        substitutions[sentence_ind, subs_inds, 0] = (
            token_indices_to_mask  # what are the token indices in the original sentence that we are masking?
        )
        substitutions[sentence_ind, subs_inds, 1] = torch.gather(
            token_tensor[sentence_ind, :], 0, token_indices_to_mask
        )  # what are the original token ids ?
        token_tensor[sentence_ind, token_indices_to_mask] = mask_id

    substitutions[:, :, 2] = -1
    substitutions[:, :, 3] = -1
    return token_tensor, attention_tensor, substitutions


def unmask_batch(
    masked_token_tensor: torch.LongTensor,
    attention_tensor: torch.Tensor,
    substitutions: torch.LongTensor,
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    rng: torch.Generator,
    substitution_step: int,
):
    """
    Unmasks  single random token from each sentencei n the batch, updating hte masked_token_tensor in place and updating the substitution tensor.

    Args:
        masked_token_tensor (torch.LongTensor): Batched token tensor.
        attention_tensor (torch.Tensor): Attention mask for the pipeline.
        substitutions (torch.LongTensor): substitutions performed so far.
        pipeline (transformers.pipelines.fill_mask.FillMaskPipeline): unmasking pipeline.
        rng (torch.Generator): Random number generator for choosing the mask token on which to operate.
        substitution_step (int): What step in the substitution chain are we unmasking -- this is noted in substitutions(:,unmasked_index, 3).
    """
    logits = pipeline.model.forward(masked_token_tensor, attention_tensor)["logits"]
    batch_size = masked_token_tensor.shape[0]
    for sent_ind in range(batch_size):
        # print('starting sentence: ', sent_ind)
        masked_token_sub_inds = torch.nonzero(
            (substitutions[sent_ind, :, 2] == -1) & (substitutions[sent_ind, :, 0] >= 0)
        )
        # print('mask of permitted substitutions: ', (substitutions[sent_ind, :, 2] == -1) & (substitutions[sent_ind,:,0] >= 0))
        # print('masked token sub inds: ',masked_token_sub_inds)
        if masked_token_sub_inds.shape[0] == 0:
            # then, there are no masked tokens remaining in the sentence, and we should continue with another sentence.
            # print(sent_ind, "skipping sentence!")
            continue
        unmask_index = masked_token_sub_inds[
            torch.randint(
                0,
                masked_token_sub_inds.shape[0],
                (1,),
                generator=rng,
                device=rng.device,
            )
        ]
        # print('unmasking token: ',unmask_index, )
        token_index_in_sent = substitutions[sent_ind, unmask_index, 0]
        new_token_pmf = (
            logits[sent_ind, token_index_in_sent, :].squeeze().softmax(0)
        )  # probability mass function of new tokens.
        new_token_id = torch.multinomial(
            new_token_pmf, 1, False, generator=rng
        )  # sampling a single token
        masked_token_tensor[sent_ind, token_index_in_sent] = substitutions[
            sent_ind, unmask_index, 2
        ] = new_token_id  # performing the substitution
        substitutions[sent_ind, unmask_index, 3] = substitution_step


def apply_substitutions(
    token_tensor: torch.LongTensor, substitutions: torch.LongTensor, state="final"
) -> None:
    """Applies the mask-unmask substitutions to a token tensor, for instance to see the final text.

    Args:
        token_tensor (torch.LongTensor): The token tensor to be transformed, representing the initial input sentences.
        substitutions (torch.LongTensor): The substitution record tensor
        state (str): One of 'final' or 'original' -- whether to restor the token tensor to the original state, or to apply the given substitutions.
    """
    assert (
        token_tensor.shape[0] == substitutions.shape[0]
    )  # ensure the batch sizes are the same.
    assert state in {"final", "original"}
    substitution_index = 2 if state == "final" else 1

    batch_indices = torch.arange(
        token_tensor.shape[0], device=token_tensor.device
    ).unsqueeze(1)
    token_tensor[batch_indices, substitutions[:, :, 0]] = substitutions[
        :, :, substitution_index
    ]

    # batch_size = token_tensor.shape[0]
    # for sent_ind in range(batch_size):
    #     token_tensor[sent_ind,substitutions[sent_ind,:,0]] = substitutions[sent_ind,:,substitution_index]


def mask_unmask_monte_batch(
    texts: list[str],
    pipeline: transformers.pipelines.fill_mask.FillMaskPipeline,
    num_masks: Union[int | float],
    rng: torch.Generator,
    return_tokens: bool = False,
) -> Union[torch.LongTensor, tuple[torch.LongTensor, torch.LongTensor]]:
    """
    Runs a mask-unmask monte carlo experiment on a set of texts using a fill mask pipeline.

    Args:
        texts (list[str]): The set of texts on which to act.
        pipeline (transformers.pipelines.fill_mask.FillMaskPipeline): The fill mask pipeline.
        num_masks (Union[int,  float]): Number of mask tokens to add to each text, or (if float) a probability between 0 and 1 for masking each token.
        rng (torch.Generator): The random number generator used to perform masking and to choose unmasked characters.
        return_tokens (bool): Return the token tensor as well.

    Returns:
        torch.LongTensor: The substitutions tensor, of shape [batch_size = len(texts), num_masks, 4 ].
        Last index is:
          0: masked token position in sentence (-1 indicates no masking was needed due to batching),
          1: original token id,
          2: replacement token id,
          3: unmasking step number at time of unmasking.
    """
    masked_token_tensor, attention_tensor, substitutions = prepare_masked_batch(
        texts, num_masks, rng, pipeline.tokenizer, pipeline.device
    )
    maximum_number_of_masks = substitutions.shape[1]

    for substitution_step in range(maximum_number_of_masks):
        unmask_batch(
            masked_token_tensor,
            attention_tensor,
            substitutions,
            pipeline,
            rng,
            substitution_step,
        )

    if return_tokens:
        return substitutions, masked_token_tensor
    else:
        # is there any point saving the original masked token tensor? No. We can reconstruct it from the substitution tensor and by tokenizing the orignal sentences.
        return substitutions


def mask_all_single(
    text: str, pipeline: transformers.pipelines.fill_mask.FillMaskPipeline
) -> tuple[torch.LongTensor, torch.DoubleTensor]:
    """
    Masks each token in the text, then performs inference on that masked token.

    Args:
        text (str): The source text for masking.
        pipeline (transformers.pipelines.fill_mask.FillMaskPipeline): The unmasking pipeline.

    Returns:
        torch.LongTensor: The token ids that were masked in the sentence.
        torch.DoubleTensor: The output logits from each masked token.
    """
    tokenizer = pipeline.tokenizer
    mask_id = tokenizer.mask_token_id
    tokenized_text = tokenizer(text, return_tensors="pt")
    # sending the token tensors to the appropriate device.
    tokens = tokenized_text["input_ids"].to(pipeline.device)
    # here, we duplicate the tokenized text N-2 times, because we don't want to mask the start and end of sentence tokens.
    tokenized_replicates = tokens.reshape(1, -1).repeat(tokens.size(1) - 2, 1)
    # masking along the diagonal:
    tokenized_replicates[
        torch.arange(0, tokenized_replicates.shape[0]),
        torch.arange(1, tokens.size(1) - 1),
    ] = mask_id
    # building the attention mask after sending the attention mask to the appropriate device.
    att = tokenized_text["attention_mask"].to(pipeline.device)
    attention_replicates = att.expand(tokenized_replicates.shape)
    # computing the logits for the masked tokens:
    print('\ntokenized replicates shape: ', tokenized_replicates.shape)
    with torch.no_grad():
        logits = pipeline.model.forward(tokenized_replicates, attention_replicates)["logits"]
    masked_logits = logits[
        torch.arange(0, tokenized_replicates.shape[0]),
        torch.arange(1, tokens.size(1) - 1),
        :,
    ]
    return tokens[0, 1:-1], masked_logits
