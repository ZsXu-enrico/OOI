# 模型权重和依赖设置指南

## 目录结构

新服务器上应该有这样的结构：

```
workspace/
├── TRELLIS/                           # TRELLIS主代码（已有）
├── MV-Adapter/                        # MVAdapter代码（已clone）
├── 3DEnhancer/                        # 3DEnhancer代码
│   └── src/
│
├── final/                             # 你的工作目录
│   ├── batch_image_gaussian.py
│   ├── generate_flux_edit.py
│   ├── batch_mv.py
│   ├── batch_recon3.py
│   ├── batch_refine.py
│   ├── human_data/
│   ├── animal_data/
│   ├── object_data/
│   └── mesh_gaussian/
│
└── pretrained_models/                 # 所有模型权重统一放这里
    ├── MVAdapter/
    │   ├── stable-diffusion-xl-base-1.0/
    │   ├── sdxl-vae-fp16-fix/
    │   └── mv-adapter/
    │       └── mvadapter_i2mv_sdxl.safetensors
    ├── FLUX.1-Kontext-dev/
    ├── 3DEnhancer/
    │   └── model.safetensors
    ├── grounding-dino-base/
    └── pixart_sigma_sdxlvae_T5_diffusers/
```

---

## 需要下载的模型

### 1. 自动下载的模型（首次运行时自动下载）

这些模型会在首次运行时自动从Hugging Face下载到 `~/.cache/huggingface/`：

✅ **TRELLIS** (约5GB)
```bash
# 会自动下载: JeffreyXiang/TRELLIS-image-large
# 首次运行 batch_image_gaussian.py 时自动下载
```

✅ **SAM2** (约900MB)
```bash
# 会自动下载: facebook/sam2.1-hiera-large
# 首次运行 batch_mv.py 时自动下载
```

✅ **VGG-T** (约4GB) - 可选，仅 batch_recon3.py 使用
```bash
# 会自动下载: facebook/VGGT-1B
# 如果不需要depth loss可以跳过
```

---

### 2. 需要手动下载的模型

#### 2.1 MVAdapter 相关 (必需 - batch_mv.py使用)

**stable-diffusion-xl-base-1.0** (约7GB)
```bash
# 下载地址: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
cd pretrained_models/MVAdapter
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
```

**sdxl-vae-fp16-fix** (约300MB)
```bash
# 下载地址: https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
cd pretrained_models/MVAdapter
git clone https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
```

**mv-adapter 权重** (约1GB)
```bash
# 下载地址: https://huggingface.co/huanngzh/mv-adapter
cd pretrained_models/MVAdapter
mkdir -p mv-adapter
cd mv-adapter
wget https://huggingface.co/huanngzh/mv-adapter/resolve/main/mvadapter_i2mv_sdxl.safetensors
```

#### 2.2 FLUX Kontext (必需 - generate_flux_edit.py使用)

**FLUX.1-Kontext-dev** (约24GB)
```bash
# 下载地址: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
cd pretrained_models
git lfs install
git clone https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
```

#### 2.3 Grounding DINO (必需 - batch_mv.py的分割功能)

**grounding-dino-base** (约700MB)
```bash
# 下载地址: https://huggingface.co/IDEA-Research/grounding-dino-base
cd pretrained_models
git clone https://huggingface.co/IDEA-Research/grounding-dino-base
```

#### 2.4 3DEnhancer (可选 - 仅batch_refine.py使用SDS时需要)

**3DEnhancer model** (约5GB)
```bash
# 下载地址: https://huggingface.co/flamehaze1115/3DEnhancer
cd pretrained_models
mkdir -p 3DEnhancer
cd 3DEnhancer
wget https://huggingface.co/flamehaze1115/3DEnhancer/resolve/main/model.safetensors
```

**PixArt-Sigma** (约5GB) - 3DEnhancer依赖
```bash
# 下载地址: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
cd pretrained_models
git clone https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers
```

**3DEnhancer 代码**
```bash
cd workspace
git clone https://github.com/flamehaze1115/3DEnhancer.git
```

---

## 修改配置文件

需要修改 `final/` 目录下脚本中的硬编码路径：

### batch_mv.py
```python
# 第153-155行，修改为：
base_model = "../pretrained_models/MVAdapter/stable-diffusion-xl-base-1.0"
vae_model = "../pretrained_models/MVAdapter/sdxl-vae-fp16-fix"
adapter_path = "../pretrained_models/MVAdapter/mv-adapter"

# 第285行，修改为：
ground_dino_model_path = "../pretrained_models/grounding-dino-base"
```

### generate_flux_edit.py
```python
# 第48行，修改为：
flux_model_path: str = "../pretrained_models/FLUX.1-Kontext-dev"
```

### batch_refine.py
```python
# 第72-74行，修改为：
self.enhancer_model_path = "../pretrained_models/3DEnhancer/model.safetensors"
self.enhancer_config_path = "../3DEnhancer/src/configs/config.py"
self.pixart_pretrained_path = "../pretrained_models/pixart_sigma_sdxlvae_T5_diffusers"
```

---

## 快速开始

### 最小配置（不使用MVAdapter和SDS）

如果你只想快速测试基础功能，可以只下载：

```bash
# 不需要手动下载！
# 只运行 batch_image_gaussian.py
python batch_image_gaussian.py --data_type human

# TRELLIS会自动从HuggingFace下载
```

这样可以生成拓扑高斯，但无法运行后续步骤。

---

### 完整配置（包含所有功能）

需要下载所有模型（约50GB）：

1. **MVAdapter相关** (~8.3GB)
2. **FLUX Kontext** (~24GB)
3. **Grounding DINO** (~700MB)
4. **3DEnhancer** (~10GB)

总共约 **43GB** 硬盘空间（不包含自动下载的模型）

---

## 验证安装

```bash
# 1. 检查目录结构
ls -la pretrained_models/MVAdapter/
ls -la pretrained_models/FLUX.1-Kontext-dev/
ls -la pretrained_models/grounding-dino-base/

# 2. 测试TRELLIS（自动下载）
cd final
python batch_image_gaussian.py --data_type human --start_idx 0 --end_idx 1

# 3. 测试MVAdapter
python batch_mv.py --use_mvadapter

# 4. 测试完整流程
# 先运行batch_image_gaussian, 再generate_flux_edit, 然后batch_mv, batch_recon3, batch_refine
```

---

## 常见问题

### Q: 某些模型下载很慢？
A: 使用镜像或代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或使用代理
export HTTP_PROXY=http://proxy:port
```

### Q: 磁盘空间不够？
A: 可以只下载必需模型：
- **batch_image_gaussian**: 只需TRELLIS（自动下载）
- **generate_flux_edit**: 需要FLUX Kontext
- **batch_mv**: 需要MVAdapter + Grounding DINO
- **batch_recon3**: 不需要额外模型
- **batch_refine**: 需要3DEnhancer（仅--use_sds时）

### Q: Git LFS下载失败？
A: 手动下载大文件：
```bash
# 进入repo目录
cd pretrained_models/stable-diffusion-xl-base-1.0
git lfs pull
# 或者用wget下载.safetensors文件
```

---

## 总结

**最简安装**（只生成拓扑高斯）：
- ✅ 无需手动下载，TRELLIS自动下载

**标准安装**（生成+编辑+多视角+重建）：
- MVAdapter相关 (~8.3GB)
- FLUX Kontext (~24GB)
- Grounding DINO (~700MB)
- **总计：~33GB**

**完整安装**（包含SDS精细化）：
- 标准安装 + 3DEnhancer (~10GB)
- **总计：~43GB**