import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils


def causal_attention_mask(seq_len: int):
    return jnp.tril(jnp.ones((seq_len, seq_len)))


class TransformerBlock(nnx.Module):

    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, mesh,
                 rngs: nnx.Rngs, rate: float = 0.1):
        self.mha = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.xavier_uniform(),
                NamedSharding(mesh, P(None, 'model'))
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros_init(),
                NamedSharding(mesh, P('model'))
            ),
            rngs=rngs
        )

        self.dropout1 = nnx.Dropout(rate=rate)
        self.layer_norm1 = nnx.LayerNorm(epsilon=1e-6, num_features=embed_dim,
                                         scale_init=nnx.with_partitioning(
                                             nnx.initializers.xavier_unior
                                         ))
