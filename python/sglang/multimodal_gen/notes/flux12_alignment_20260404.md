# FLUX.1 / FLUX.2 对齐记录 2026-04-04

## 基线

- 本地 worktree: `/Users/mick/repos/sglang/.claude/worktrees/flux12-align-pr22059`
- worktree 初始基线: PR `#22059` head `047ad94efc277ea850700355af9f3b8c0a13a686`
- 当前修复 commit:
  - `491589399864bf75a9cfa253a3f21095e97465cf`
- 远端执行环境:
  - host: `sglang-diffusion@124.158.103.3` (`ac-h200-gpu03`)
  - container: `sglang_mick`
  - repo: `/sgl-workspace/sglang`
- 远端三验:
  - `git rev-parse HEAD = 491589399864bf75a9cfa253a3f21095e97465cf`
  - `component_loader.py sha1 = bdc17a3ea58d039a41006f5254e8c3b6e6ebdaaf`
  - `mistral_3.py sha1 = 0defb0266d97a47a40722b7f8833080f8ff014fe`

## 本次修改

- `runtime/loader/component_loaders/component_loader.py`
  - 恢复通用 tokenizer loader 为 `padding_side="right", use_fast=True`
  - 仅对 `Flux2PipelineConfig` 特判回到 `AutoTokenizer.from_pretrained(component_model_path)` 默认行为
- `runtime/models/encoders/mistral_3.py`
  - `output_hidden_states=True` 时，改为按 HF 语义收集 hidden states:
    - 每层前 append 输入 hidden states
    - 最后 append norm 后的 hidden states

## 精度进度

- FLUX.2
  - 已完全对齐到 `text encoder -> transformer -> vae`
  - `test_accuracy_1_gpu_a.py -k flux_2_image_t2i`
    - VAE: `CosSim=0.999991`
    - Transformer: `CosSim=0.991878`
    - Text encoder: `CosSim=0.999987`
  - 结论:
    - FLUX.2 的文本链问题由 tokenizer loader + Mistral3 hidden states 顺序共同修复
    - 当前 server accuracy case 已通过

- FLUX.1
  - 已对齐到 `transformer -> vae`
  - `test_accuracy_1_gpu_a.py -k flux_image_t2i`
    - VAE: `CosSim=0.999993`
    - Transformer: `CosSim=0.999997`
  - 说明:
    - 初次失败不是精度问题，而是远端 `FLUX.1-dev` cache 不完整
    - 清理该模型 cache 后，重新下载并复跑通过

- FLUX.1 text encoder 额外 probe
  - 手动绕过 `accuracy_config.py` 里的 skip，直接调用 `run_text_encoder_accuracy_case`
  - 结果: `flux_image_t2i_encoder CosSim=0.466837`
  - 与现有 skip 记录一致
  - 当前判断:
    - 这个 probe 测的是 `ComponentType.TEXT_ENCODER` 的第一路 encoder（`text_encoder`，优先于 `text_encoder_2`）
    - FLUX.1 实际生成主链所依赖的条件输入已经通过 transformer accuracy 间接验证
    - 因此本次用户目标里的 T2I server testcase 精度已经到位，但 `flux_image_t2i` 的独立 text encoder component skip 仍未消除

## 结论

- 当前已经完成:
  - FLUX.2 server T2I testcase 的 native-vs-diffusers 精度对齐
  - FLUX.1 server T2I testcase 的生成相关主链对齐（VAE + transformer）
- 当前未完成:
  - `accuracy_config.py` 中 `flux_image_t2i` 的 text encoder skip 还不能删除
  - 如需继续消除此 skip，下一步应专门分析 FLUX.1 第一文本编码器 component probe 与生成主链之间的语义不一致点
