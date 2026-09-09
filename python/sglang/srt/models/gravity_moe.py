"""
GravityMoE model implementation for SGLang.

GravityMoE shares the same architecture as DeepSeek V3.
"""

from sglang.srt.models.deepseek_v2 import (
    DeepseekV2AttentionMLA,
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
    DeepseekV2MLP,
    DeepseekV2MoE,
    DeepseekV2Model,
    MoEGate,
)

# Re-export architecture components under GravityMoE naming
GravityMoEMLP = DeepseekV2MLP
GravityMoEGate = MoEGate
GravityMoEMoE = DeepseekV2MoE
GravityMoEAttentionMLA = DeepseekV2AttentionMLA
GravityMoEDecoderLayer = DeepseekV2DecoderLayer
GravityMoEModel = DeepseekV2Model


class GravityMoEForCausalLM(DeepseekV2ForCausalLM):
    pass


EntryClass = GravityMoEForCausalLM
