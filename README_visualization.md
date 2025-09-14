# 模型测试结果可视化工具

这个工具可以将模型测试结果JSON文件转换为交互式HTML可视化页面，支持通过HTTP服务器访问。

## 功能特性

- 🎨 **美观的界面设计** - 现代化的渐变背景和卡片式布局
- 📊 **实时统计信息** - 显示总测试数、成功率、平均生成时间等
- 🔍 **多维度过滤** - 按模型、提示词、步数、CFG值进行筛选
- 📈 **性能对比表格** - 详细展示各模型的性能指标
- 📊 **可视化图表** - 生成时间分布柱状图
- 🖼️ **图像网格展示** - 直观查看生成的测试图像
- 🌐 **HTTP服务器** - 支持通过浏览器访问

## 文件说明

- `generate_visualization.py` - 主要脚本，生成HTML可视化页面
- `serve_visualization.py` - 快速启动HTTP服务器的简化脚本
- `port_manager.py` - 端口管理工具，查找和杀死占用端口的进程
- `model_test_visualization.html` - 生成的HTML可视化页面

## 使用方法

### 1. 生成HTML可视化页面

```bash
# 基本用法
python generate_visualization.py test_results.json

# 指定输出文件
python generate_visualization.py test_results.json -o my_visualization.html

# 生成并启动HTTP服务器
python generate_visualization.py test_results.json --serve 8080
```

### 2. 启动HTTP服务器访问

```bash
# 自动查找HTML文件并启动服务器
python serve_visualization.py

# 指定端口
python serve_visualization.py -p 8080

# 指定HTML文件
python serve_visualization.py -f visualization.html -p 8080

# 指定目录
python serve_visualization.py -d /path/to/html/directory -p 8080
```

### 3. 命令行参数说明

#### generate_visualization.py

- `json_file` - 输入的JSON测试结果文件路径（必需）
- `-o, --output` - 输出的HTML文件路径
- `--serve PORT` - 启动HTTP服务器在指定端口

#### serve_visualization.py

- `-f, --file` - HTML文件路径
- `-d, --directory` - HTML文件所在目录
- `-p, --port` - 服务器端口（默认：8080）
- `--host` - 服务器主机地址（默认：localhost）

#### port_manager.py

- `ports` - 要检查的端口号列表（必需）
- `--kill` - 杀死占用端口的进程
- `--force` - 强制杀死进程（使用SIGKILL信号）
- `--info` - 显示进程详细信息

## 使用示例

### 示例1：生成可视化页面

```bash
cd /mnt/hz_trainer
python generate_visualization.py output/simple_test/test_results.json
```

### 示例2：生成并启动服务器

```bash
cd /mnt/hz_trainer
python generate_visualization.py output/simple_test/test_results.json --serve 8080
```

### 示例3：快速启动服务器

```bash
cd /mnt/hz_trainer/output/simple_test
python ../../serve_visualization.py -p 8080
```

### 示例4：端口管理

```bash
# 查看端口8080的占用情况
python port_manager.py 8080

# 杀死占用端口8080的进程
python port_manager.py 8080 --kill

# 强制杀死占用端口8080的进程
python port_manager.py 8080 --force

# 查看进程详细信息
python port_manager.py 8080 --info
```

## 可视化页面功能

### 统计信息卡片
- 总测试数
- 成功率
- 平均生成时间
- 测试模型数

### 过滤器
- 模型选择
- 提示词筛选
- 步数选择
- CFG值选择

### 性能对比表格
- 模型名称
- 测试数量
- 成功率
- 平均生成时间
- 最快/最慢生成时间

### 时间分布图表
- 各模型平均生成时间的柱状图对比

### 图像网格
- 显示所有测试生成的图像
- 悬停显示详细信息
- 支持图像缩放效果

## 技术特性

- **响应式设计** - 支持桌面和移动设备
- **CORS支持** - 允许跨域访问
- **手动访问** - 服务器启动后显示访问地址，需要手动打开浏览器
- **错误处理** - 完善的错误提示和处理
- **端口冲突检测** - 自动检测端口占用情况
- **优雅关闭** - 支持信号处理和优雅关闭服务器
- **端口管理** - 提供端口管理工具，方便查找和杀死占用端口的进程

## 注意事项

1. 确保JSON文件格式正确，包含必要的测试数据
2. 图像文件路径需要正确，否则会显示"图像未找到"
3. 如果端口被占用，请尝试其他端口号
4. 服务器启动后按 Ctrl+C 停止

## 故障排除

### 常见问题

1. **JSON文件加载失败**
   - 检查文件路径是否正确
   - 确认JSON格式是否有效

2. **端口被占用**
   - 尝试使用其他端口号
   - 使用端口管理工具：`python port_manager.py 8080 --kill`
   - 检查是否有其他服务占用该端口

3. **图像无法显示**
   - 检查图像文件路径是否正确
   - 确认图像文件是否存在

4. **无法访问页面**
   - 检查服务器是否正常启动
   - 确认端口号是否正确
   - 检查防火墙设置

## 开发说明

这个工具参考了 [ComfyUI Browser](https://github.com/talesofai/comfyui-browser) 的设计理念，提供了类似的交互式可视化体验。

主要技术栈：
- HTML5 + CSS3 + JavaScript
- Python HTTP服务器
- JSON数据处理
- 响应式Web设计 