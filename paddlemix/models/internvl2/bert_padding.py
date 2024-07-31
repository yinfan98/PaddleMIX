import sys
sys.path.append('/hy-tmp/test/utils')
import paddle_aux
import paddle
from einops import rearrange, repeat


class IndexFirstAxis(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = tuple(input.shape)[0], tuple(input
            .shape)[1:]
        second_dim = other_shape.size
        return paddle.take_along_axis(arr=rearrange(input,
            'b ... -> b (...)'), axis=0, indices=repeat(indices, 'z -> z d',
            d=second_dim)).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        """Class Attribute: torch.autograd.function.FunctionCtx.saved_tensors, can not convert, please check whether it is torch.Tensor.*/torch.autograd.function.FunctionCtx.*/torch.distributions.Distribution.* and convert manually"""
        (indices,) = ctx.saved_tensor()
        assert grad_output.ndim >= 2
        other_shape = tuple(grad_output.shape)[1:]
        grad_output = rearrange(grad_output, 'b ... -> b (...)')
        grad_input = paddle.zeros(shape=[ctx.first_axis_dim, tuple(
            grad_output.shape)[1]], dtype=grad_output.dtype)
        grad_input.put_along_axis_(axis=0, indices=repeat(indices,
            'z -> z d', d=tuple(grad_output.shape)[1]), values=grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = paddle.zeros(shape=[first_axis_dim, *tuple(values.shape)[1
            :]], dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Class Attribute: torch.autograd.function.FunctionCtx.saved_tensors, can not convert, please check whether it is torch.Tensor.*/torch.autograd.function.FunctionCtx.*/torch.distributions.Distribution.* and convert manually"""
        (indices,) = ctx.saved_tensor()
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


class IndexFirstAxisResidual(paddle.autograd.PyLayer):

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = tuple(input.shape)[0], tuple(input
            .shape)[1:]
        second_dim = other_shape.size
        output = input[indices]
        return output, input.detach()

    @staticmethod
    def backward(ctx, grad_output, grad_residual):
        """Class Attribute: torch.autograd.function.FunctionCtx.saved_tensors, can not convert, please check whether it is torch.Tensor.*/torch.autograd.function.FunctionCtx.*/torch.distributions.Distribution.* and convert manually"""
        (indices,) = ctx.saved_tensor()
        assert grad_output.ndim >= 2
        other_shape = tuple(grad_output.shape)[1:]
        assert tuple(grad_residual.shape)[1:] == other_shape
        grad_input = grad_residual
        indices = indices.reshape(tuple(indices.shape)[0], *((1,) * (
            grad_output.ndim - 1)))
        indices = indices.expand_as(y=grad_output)
        grad_input.put_along_axis_(axis=0, indices=indices, values=
            grad_output, reduce='add')
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis_residual = IndexFirstAxisResidual.apply


def unpad_input(hidden_states, attention_mask):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    seqlens_in_batch = attention_mask.sum(axis=-1, dtype='int32')
    indices = paddle.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = paddle_aux._FUNCTIONAL_PAD(pad=(1, 0), x=paddle.cumsum(x=
        seqlens_in_batch, axis=0, dtype='int32'))
    return index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'
        ), indices), indices, cu_seqlens, max_seqlen_in_batch


def unpad_input_for_concatenated_sequences(hidden_states,
    attention_mask_in_length):
    """
    Supports concatenating short samples in one sequence. The attention_mask_in_length is utilized to mask other short samples. It helps efficient training of variant lengths-based samples (e.g., the supervised fine-tuning task in large language model).
    The motivation for this function is explained [here](https://github.com/Dao-AILab/flash-attention/issues/432#issuecomment-1668822286).
    
    For example, if batch = 3 and seqlen = 6, the attention_mask_in_length is:
        ```
        [
          [2, 3, 0, 0, 0, 0],
          [3, 2, 0, 0, 0, 0],
          [6, 0, 0, 0, 0, 0]
        ]
        ```
    , which refers to the 3D-attention mask:
        ```
        [
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1]
          ],
          [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1]
          ]
        ]
        ```.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask_in_length: (batch, seqlen), int, a nonzero number (e.g., 1, 2, 3, etc.) means length of concatenated sequence in b-th batch, and 0 means none.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices of non-masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
    """
    length = attention_mask_in_length.sum(axis=-1)
    seqlen = attention_mask_in_length.shape[-1]
    attention_mask_2d = paddle.arange(dtype=length.dtype, end=seqlen).expand(
        shape=[len(length), seqlen]) < length.unsqueeze(axis=1)
    real_indices_idx = paddle.nonzero(attention_mask_in_length.flatten(),
        as_tuple=False).flatten()
    seqlens_in_batch = attention_mask_in_length.flatten()[real_indices_idx]
    indices = paddle.nonzero(attention_mask_2d.flatten(), as_tuple=False
        ).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = paddle_aux._FUNCTIONAL_PAD(pad=(1, 0), x=paddle.cumsum(x=
        seqlens_in_batch, axis=0, dtype='int32'))
    return index_first_axis(rearrange(hidden_states, 'b s ... -> (b s) ...'
        ), indices), indices, cu_seqlens, max_seqlen_in_batch


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = tuple(hidden_states.shape)[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, '(b s) ... -> b s ...', b=batch)
