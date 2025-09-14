#!/usr/bin/env python3
"""
生成模型测试结果可视化HTML页面
"""

import json
import os
import sys
import argparse
from pathlib import Path

def load_test_results(json_path):
    """加载测试结果JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载JSON文件失败: {e}")
        return None

def generate_html_content(test_data):
    """生成HTML内容"""
    html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>模型测试结果可视化</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }

        .stat-card:hover {
            transform: translateY(-5px);
        }

        .stat-number {
            font-size: 2.5em;
            font-weight: bold;
            color: #4facfe;
            margin-bottom: 5px;
        }

        .stat-label {
            color: #666;
            font-size: 0.9em;
        }

        .controls {
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #eee;
        }

        .filter-group {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            align-items: center;
        }

        .filter-item {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .filter-item label {
            font-weight: bold;
            color: #333;
            font-size: 0.9em;
        }

        .filter-item select, .filter-item input {
            padding: 8px 12px;
            border: 2px solid #e1e5e9;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s ease;
        }

        .filter-item select:focus, .filter-item input:focus {
            outline: none;
            border-color: #4facfe;
        }

        .image-grid {
            padding: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }

        .image-card {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
        }

        .image-container {
            position: relative;
            width: 100%;
            height: 300px;
            overflow: hidden;
            background: #f0f0f0;
        }

        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.3s ease;
        }

        .image-card:hover .image-container img {
            transform: scale(1.05);
        }

        .image-info {
            padding: 15px;
        }

        .image-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            font-size: 1.1em;
        }

        .image-details {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            font-size: 0.85em;
            color: #666;
        }

        .detail-item {
            display: flex;
            justify-content: space-between;
        }

        .detail-label {
            font-weight: 500;
        }

        .detail-value {
            color: #4facfe;
            font-weight: bold;
        }

        .performance-chart {
            padding: 30px;
            background: white;
            margin: 20px 30px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .chart-title {
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 20px;
            color: #333;
            text-align: center;
        }

        .chart-container {
            height: 400px;
            position: relative;
        }

        .loading {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 200px;
            font-size: 1.2em;
            color: #666;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #4facfe;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin-right: 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .no-results {
            text-align: center;
            padding: 50px;
            color: #666;
            font-size: 1.2em;
        }

        .model-comparison {
            padding: 30px;
            background: white;
            margin: 20px 30px;
            border-radius: 12px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .comparison-table th,
        .comparison-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }

        .comparison-table th {
            background: #f8f9fa;
            font-weight: bold;
            color: #333;
        }

        .comparison-table tr:hover {
            background: #f8f9fa;
        }

        .success-rate {
            color: #28a745;
            font-weight: bold;
        }

        .avg-time {
            color: #ffc107;
            font-weight: bold;
        }

        @media (max-width: 768px) {
            .container {
                margin: 10px;
                border-radius: 10px;
            }

            .header {
                padding: 20px;
            }

            .header h1 {
                font-size: 2em;
            }

            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
                padding: 20px;
            }

            .filter-group {
                flex-direction: column;
                align-items: stretch;
            }

            .image-grid {
                grid-template-columns: 1fr;
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 模型测试结果可视化</h1>
            <p>AI图像生成模型性能对比分析</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="total-tests">-</div>
                <div class="stat-label">总测试数</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="success-rate">-</div>
                <div class="stat-label">成功率</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="avg-time">-</div>
                <div class="stat-label">平均生成时间(秒)</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="total-models">-</div>
                <div class="stat-label">测试模型数</div>
            </div>
        </div>

        <div class="controls">
            <div class="filter-group">
                <div class="filter-item">
                    <label for="model-filter">模型选择:</label>
                    <select id="model-filter">
                        <option value="">所有模型</option>
                    </select>
                </div>
                <div class="filter-item">
                    <label for="prompt-filter">提示词:</label>
                    <select id="prompt-filter">
                        <option value="">所有提示词</option>
                    </select>
                </div>
                <div class="filter-item">
                    <label for="steps-filter">步数:</label>
                    <select id="steps-filter">
                        <option value="">所有步数</option>
                    </select>
                </div>
                <div class="filter-item">
                    <label for="cfg-filter">CFG值:</label>
                    <select id="cfg-filter">
                        <option value="">所有CFG值</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="model-comparison">
            <h2 class="chart-title">📊 模型性能对比</h2>
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>模型名称</th>
                        <th>测试数量</th>
                        <th>成功率</th>
                        <th>平均生成时间</th>
                        <th>最快生成时间</th>
                        <th>最慢生成时间</th>
                    </tr>
                </thead>
                <tbody id="comparison-table-body">
                </tbody>
            </table>
        </div>

        <div class="performance-chart">
            <h2 class="chart-title">⚡ 生成时间分布</h2>
            <div class="chart-container" id="time-chart">
                <div class="loading">
                    <div class="spinner"></div>
                    加载中...
                </div>
            </div>
        </div>

        <div class="image-grid" id="image-grid">
            <div class="loading">
                <div class="spinner"></div>
                加载测试结果中...
            </div>
        </div>
    </div>

    <script>
        // 测试数据 - 直接嵌入到HTML中
        const testData = {json_data};
        let filteredData = testData.test_results;

        // 初始化页面
        function initializePage() {
            // 设置过滤器事件监听
            document.getElementById('model-filter').addEventListener('change', filterResults);
            document.getElementById('prompt-filter').addEventListener('change', filterResults);
            document.getElementById('steps-filter').addEventListener('change', filterResults);
            document.getElementById('cfg-filter').addEventListener('change', filterResults);
        }

        // 更新统计信息
        function updateStats() {
            const summary = testData.test_summary;
            const results = testData.test_results;

            // 计算平均生成时间
            const avgTime = results.reduce((sum, result) => sum + result.generation_time, 0) / results.length;

            document.getElementById('total-tests').textContent = summary.total_tests;
            document.getElementById('success-rate').textContent = summary.success_rate.toFixed(1) + '%';
            document.getElementById('avg-time').textContent = avgTime.toFixed(2);
            document.getElementById('total-models').textContent = summary.test_config.models.length;
        }

        // 更新过滤器选项
        function updateFilters() {
            const results = testData.test_results;
            
            // 获取唯一值
            const models = [...new Set(results.map(r => r.model_name))];
            const prompts = [...new Set(results.map(r => r.prompt.substring(0, 50) + '...'))];
            const steps = [...new Set(results.map(r => r.steps))].sort((a, b) => a - b);
            const cfgs = [...new Set(results.map(r => r.cfg))].sort((a, b) => a - b);

            // 填充过滤器选项
            populateSelect('model-filter', models);
            populateSelect('prompt-filter', prompts);
            populateSelect('steps-filter', steps);
            populateSelect('cfg-filter', cfgs);
        }

        function populateSelect(selectId, options) {
            const select = document.getElementById(selectId);
            const currentValue = select.value;
            
            // 保留"所有"选项
            select.innerHTML = '<option value="">所有</option>';
            
            options.forEach(option => {
                const optionElement = document.createElement('option');
                optionElement.value = option;
                optionElement.textContent = option;
                select.appendChild(optionElement);
            });
            
            select.value = currentValue;
        }

        // 过滤结果
        function filterResults() {
            const modelFilter = document.getElementById('model-filter').value;
            const promptFilter = document.getElementById('prompt-filter').value;
            const stepsFilter = document.getElementById('steps-filter').value;
            const cfgFilter = document.getElementById('cfg-filter').value;

            filteredData = testData.test_results.filter(result => {
                const modelMatch = !modelFilter || result.model_name === modelFilter;
                const promptMatch = !promptFilter || result.prompt.substring(0, 50) + '...' === promptFilter;
                const stepsMatch = !stepsFilter || result.steps.toString() === stepsFilter;
                const cfgMatch = !cfgFilter || result.cfg.toString() === cfgFilter;
                
                return modelMatch && promptMatch && stepsMatch && cfgMatch;
            });

            updateImageGrid();
            updateComparisonTable();
            updateTimeChart();
        }

        // 更新图像网格
        function updateImageGrid() {
            const grid = document.getElementById('image-grid');
            
            if (!filteredData || filteredData.length === 0) {
                grid.innerHTML = '<div class="no-results">🔍 没有找到匹配的测试结果</div>';
                return;
            }

            grid.innerHTML = filteredData.map(result => {
                // 将绝对路径转换为相对路径
                let imagePath = result.output_image;
                if (imagePath.startsWith('/mnt/hz_trainer/output/simple_test/')) {
                    imagePath = imagePath.replace('/mnt/hz_trainer/output/simple_test/', '');
                } else if (imagePath.startsWith('/')) {
                    // 如果是其他绝对路径，只取文件名
                    imagePath = imagePath.split('/').pop();
                }
                
                return `
                <div class="image-card">
                    <div class="image-container">
                        <img src="${imagePath}" alt="测试图像 ${result.test_id}" 
                             onerror="this.style.display='none'; this.parentElement.innerHTML='<div style=\\'display:flex;align-items:center;justify-content:center;height:100%;color:#999;\\'>图像未找到</div>'">
                    </div>
                    <div class="image-info">
                        <div class="image-title">测试 #${result.test_id}</div>
                        <div class="image-details">
                            <div class="detail-item">
                                <span class="detail-label">模型:</span>
                                <span class="detail-value">${result.model_name.substring(0, 20)}...</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">步数:</span>
                                <span class="detail-value">${result.steps}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">CFG:</span>
                                <span class="detail-value">${result.cfg}</span>
                            </div>
                            <div class="detail-item">
                                <span class="detail-label">时间:</span>
                                <span class="detail-value">${result.generation_time.toFixed(2)}s</span>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        }

        // 更新对比表格
        function updateComparisonTable() {
            if (!filteredData) return;

            const modelStats = {};
            
            filteredData.forEach(result => {
                if (!modelStats[result.model_name]) {
                    modelStats[result.model_name] = {
                        count: 0,
                        totalTime: 0,
                        times: [],
                        successCount: 0
                    };
                }
                
                modelStats[result.model_name].count++;
                modelStats[result.model_name].totalTime += result.generation_time;
                modelStats[result.model_name].times.push(result.generation_time);
                
                if (result.status === 'success') {
                    modelStats[result.model_name].successCount++;
                }
            });

            const tableBody = document.getElementById('comparison-table-body');
            tableBody.innerHTML = Object.entries(modelStats).map(([modelName, stats]) => {
                const avgTime = stats.totalTime / stats.count;
                const successRate = (stats.successCount / stats.count * 100).toFixed(1);
                const minTime = Math.min(...stats.times);
                const maxTime = Math.max(...stats.times);
                
                return `
                    <tr>
                        <td>${modelName}</td>
                        <td>${stats.count}</td>
                        <td class="success-rate">${successRate}%</td>
                        <td class="avg-time">${avgTime.toFixed(2)}s</td>
                        <td>${minTime.toFixed(2)}s</td>
                        <td>${maxTime.toFixed(2)}s</td>
                    </tr>
                `;
            }).join('');
        }

        // 更新时间分布图表
        function updateTimeChart() {
            if (!filteredData) return;

            const chartContainer = document.getElementById('time-chart');
            
            // 按模型分组时间数据
            const modelTimes = {};
            filteredData.forEach(result => {
                if (!modelTimes[result.model_name]) {
                    modelTimes[result.model_name] = [];
                }
                modelTimes[result.model_name].push(result.generation_time);
            });

            // 创建简单的柱状图
            const chartHTML = `
                <div style="display: flex; align-items: end; justify-content: space-around; height: 100%; padding: 20px;">
                    ${Object.entries(modelTimes).map(([modelName, times]) => {
                        const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
                        const maxTime = Math.max(...times);
                        const height = (avgTime / maxTime) * 300;
                        
                        return `
                            <div style="text-align: center;">
                                <div style="
                                    width: 60px; 
                                    height: ${height}px; 
                                    background: linear-gradient(to top, #4facfe, #00f2fe);
                                    border-radius: 5px 5px 0 0;
                                    margin-bottom: 10px;
                                    position: relative;
                                ">
                                    <div style="
                                        position: absolute;
                                        top: -25px;
                                        left: 50%;
                                        transform: translateX(-50%);
                                        background: #333;
                                        color: white;
                                        padding: 2px 6px;
                                        border-radius: 3px;
                                        font-size: 12px;
                                        white-space: nowrap;
                                    ">${avgTime.toFixed(2)}s</div>
                                </div>
                                <div style="font-size: 12px; color: #666; max-width: 80px; word-wrap: break-word;">
                                    ${modelName.substring(0, 15)}...
                                </div>
                            </div>
                        `;
                    }).join('')}
                </div>
            `;
            
            chartContainer.innerHTML = chartHTML;
        }

        // 页面加载完成后执行
        document.addEventListener('DOMContentLoaded', function() {
            initializePage();
            updateStats();
            updateFilters();
            updateComparisonTable();
            updateImageGrid();
            updateTimeChart();
        });
    </script>
</body>
</html>'''
    
    # 将JSON数据嵌入到HTML中
    json_str = json.dumps(test_data, ensure_ascii=False, indent=2)
    html_content = html_template.replace('{json_data}', json_str)
    
    return html_content

def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='生成模型测试结果可视化HTML页面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python generate_visualization.py test_results.json
  python generate_visualization.py test_results.json -o visualization.html
  python generate_visualization.py test_results.json --serve 8080
  python generate_visualization.py /path/to/test_results.json -o /path/to/output.html --serve 8080
        """
    )
    
    parser.add_argument(
        'json_file',
        help='输入的JSON测试结果文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='输出的HTML文件路径 (默认: 与JSON文件同目录下的model_test_visualization.html)'
    )
    
    parser.add_argument(
        '--serve',
        type=int,
        metavar='PORT',
        help='启动HTTP服务器在指定端口 (例如: --serve 8080)'
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置文件路径
    json_path = args.json_file
    if args.output:
        html_output_path = args.output
    else:
        # 默认输出到JSON文件同目录
        json_dir = os.path.dirname(json_path)
        html_output_path = os.path.join(json_dir, "model_test_visualization.html")
    
    print("=== 生成模型测试结果可视化页面 ===")
    print(f"📁 输入JSON文件: {json_path}")
    print(f"📁 输出HTML文件: {html_output_path}")
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ JSON文件不存在: {json_path}")
        sys.exit(1)
    
    # 加载测试数据
    print("📊 加载测试数据...")
    test_data = load_test_results(json_path)
    if test_data is None:
        print("❌ 无法加载测试数据")
        sys.exit(1)
    
    print(f"✅ 成功加载测试数据: {test_data['test_summary']['total_tests']} 个测试")
    
    # 生成HTML内容
    print("🎨 生成HTML页面...")
    html_content = generate_html_content(test_data)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(html_output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存HTML文件
    print(f"💾 保存HTML文件...")
    try:
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print("✅ HTML可视化页面生成完成!")
        print(f"📁 文件位置: {html_output_path}")
    except Exception as e:
        print(f"❌ 保存HTML文件失败: {e}")
        sys.exit(1)
    
    # 如果指定了端口，启动HTTP服务器
    if args.serve:
        start_http_server(html_output_path, args.serve)
    else:
        print("🌐 请在浏览器中打开该HTML文件查看可视化结果")
        print("💡 提示: 使用 --serve PORT 参数可以启动HTTP服务器")

def start_http_server(html_file_path, port):
    """启动HTTP服务器"""
    import http.server
    import socketserver
    import threading
    import signal
    import sys
    from urllib.parse import urljoin
    
    # 切换到HTML文件所在目录
    html_dir = os.path.dirname(os.path.abspath(html_file_path))
    html_filename = os.path.basename(html_file_path)
    
    # 全局变量存储服务器实例
    httpd = None
    
    def signal_handler(signum, frame):
        """信号处理函数"""
        nonlocal httpd
        print(f"\n🛑 收到信号 {signum}，正在关闭服务器...")
        if httpd:
            try:
                httpd.shutdown()
                httpd.server_close()
                print("✅ 服务器已正常关闭")
            except Exception as e:
                print(f"⚠️  关闭服务器时出现错误: {e}")
        # 移除sys.exit(0)，避免重复调用
        return
    
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            # 添加CORS头，允许跨域访问
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            super().end_headers()
        
        def log_message(self, format, *args):
            # 自定义日志格式
            print(f"🌐 HTTP请求: {format % args}")
    
    try:
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
        
        # 切换到HTML文件目录
        os.chdir(html_dir)
        
        # 创建服务器，监听所有网络接口
        httpd = socketserver.TCPServer(("0.0.0.0", port), CustomHTTPRequestHandler)
        
        # 设置服务器选项，允许端口重用
        httpd.allow_reuse_address = True
        
        print(f"🚀 HTTP服务器启动成功!")
        print(f"📡 服务器地址: http://localhost:{port}")
        print(f"📄 可视化页面: http://localhost:{port}/{html_filename}")
        print(f"📁 服务目录: {html_dir}")
        print("⏹️  按 Ctrl+C 停止服务器")
        print("🌐 请在浏览器中手动访问上述地址")
        
        # 启动服务器
        httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ 端口 {port} 已被占用，请尝试其他端口")
            print("💡 提示: 可以使用以下命令查找占用端口的进程:")
            print(f"   lsof -i :{port}")
            print(f"   netstat -tulpn | grep :{port}")
        else:
            print(f"❌ 启动HTTP服务器失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  收到中断信号，正在关闭服务器...")
        if httpd:
            try:
                httpd.shutdown()
                httpd.server_close()
                print("✅ 服务器已正常关闭")
            except Exception as e:
                print(f"⚠️  关闭服务器时出现错误: {e}")
        return True
    except Exception as e:
        print(f"❌ HTTP服务器错误: {e}")
        if httpd:
            try:
                httpd.server_close()
            except:
                pass
        return False

if __name__ == "__main__":
    main()
