# 通用模型加载器 (ModelLoader)

这是一个通用的PyTorch模型加载器，支持多种模型格式和加载方式。

## 功能特性

- ✅ 支持多种模型格式：`.safetensors`, `.pth`, `.pt`, `.ckpt`
- ✅ 自动格式检测
- ✅ PyTorch 2.6+ 兼容性处理
- ✅ 设备管理（CPU/GPU）
- ✅ 错误处理和日志记录
- ✅ 灵活的加载选项

## 安装依赖

```bash
pip install torch safetensors
```

## 基本使用

### 1. 基本模型加载

```python
from common.model_loader import ModelLoader
from models.lumina.models import NextDiT_2B_GQA_patch2_Adaln_Refiner

# 创建模型实例
model = NextDiT_2B_GQA_patch2_Adaln_Refiner()

# 使用加载器加载权重
loader = ModelLoader(device='cuda' if torch.cuda.is_available() else 'cpu')
model = loader.load_weights("path/to/model.safetensors", model)
```

### 2. 只加载权重字典

```python
from common.model_loader import load_model_weights

# 只加载权重，不创建模型实例
weights = load_model_weights("path/to/model.safetensors")
print(f"权重键的数量: {len(weights)}")
```

### 3. 使用便捷函数

```python
from common.model_loader import load_model

# 一步完成模型创建和权重加载
model = load_model(
    model_class=NextDiT_2B_GQA_patch2_Adaln_Refiner,
    model_path="path/to/model.safetensors",
    device='cuda'
)
```

## 高级功能

### 1. 非严格加载

```python
# 允许部分权重不匹配
model = loader.load_weights("path/to/model.safetensors", model, strict=False)
```

### 2. 自定义配置

```python
config = {
    'hidden_size': 768,
    'num_layers': 12,
    # 其他模型参数...
}

model = loader.load_model_with_config(
    model_class=NextDiT_2B_GQA_patch2_Adaln_Refiner,
    model_path="path/to/model.safetensors",
    config=config
)
```

### 3. 错误处理

```python
try:
    model = loader.load_weights("path/to/model.safetensors", model)
except FileNotFoundError:
    print("模型文件不存在")
except Exception as e:
    print(f"加载失败: {e}")
```

## 支持的格式

### Safetensors (.safetensors)
- 最安全的格式，防止任意代码执行
- 需要安装 `safetensors` 库

### PyTorch (.pth, .pt, .ckpt)
- 标准PyTorch格式
- 自动处理PyTorch 2.6+的兼容性问题

## API 参考

### ModelLoader 类

#### `__init__(device=None)`
- `device`: 设备类型 ('cpu', 'cuda', 'mps' 等)

#### `load_weights(model_path, model=None, strict=True, **kwargs)`
- `model_path`: 模型文件路径
- `model`: 要加载权重的模型实例
- `strict`: 是否严格匹配权重
- 返回：模型实例或权重字典

#### `load_model_with_config(model_class, model_path, config=None, **kwargs)`
- `model_class`: 模型类
- `model_path`: 模型文件路径
- `config`: 模型配置字典
- 返回：加载了权重的模型实例

### 便捷函数

#### `load_model_weights(model_path, device=None, **kwargs)`
- 只加载权重字典

#### `load_model(model_class, model_path, config=None, device=None, **kwargs)`
- 一步完成模型创建和权重加载

## 示例

查看 `examples/model_loading_examples.py` 获取完整的使用示例。

## 注意事项

1. **安全性**: 使用 `weights_only=False` 时要注意文件来源的安全性
2. **内存**: 大模型加载时注意内存使用
3. **设备**: 确保模型和设备兼容性
4. **格式**: 优先使用 `.safetensors` 格式以提高安全性

## 故障排除

### 常见错误

1. **"Weights only load failed"**
   - 原因：PyTorch 2.6+ 默认使用 `weights_only=True`
   - 解决：加载器会自动处理，无需手动干预

2. **"safetensors not available"**
   - 原因：未安装 safetensors 库
   - 解决：`pip install safetensors`

3. **"模型文件不存在"**
   - 原因：文件路径错误
   - 解决：检查文件路径是否正确

4. **"无法识别文件格式"**
   - 原因：文件格式不支持或损坏
   - 解决：检查文件格式和完整性 