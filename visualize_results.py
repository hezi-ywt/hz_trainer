#!/usr/bin/env python3
"""
测试结果可视化脚本
根据JSON文件生成HTML报告
"""

import json
import os
import sys
from datetime import datetime

def create_html_report(json_file_path, output_html_path):
    """创建HTML报告"""
    
    # 读取JSON数据
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    test_summary = data['test_summary']
    test_results = data['test_results']
    
    # 生成HTML内容
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lumina2模型测试结果报告</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .summary-section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .summary-card {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .summary-card h3 {{
            font-size: 2em;
            margin-bottom: 5px;
        }}
        
        .summary-card p {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .config-section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .config-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .config-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #007bff;
        }}
        
        .config-item h4 {{
            color: #007bff;
            margin-bottom: 10px;
        }}
        
        .results-section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .filters {{
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .filter-group label {{
            font-weight: bold;
            color: #555;
        }}
        
        .filter-group select, .filter-group input {{
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }}
        
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .result-card {{
            background: #f8f9fa;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}
        
        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }}
        
        .result-image {{
            width: 100%;
            height: 200px;
            background: #e9ecef;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #6c757d;
            font-size: 14px;
        }}
        
        .result-image img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
        }}
        
        .result-info {{
            padding: 15px;
        }}
        
        .result-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        
        .test-id {{
            background: #007bff;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .status {{
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        
        .status.success {{
            background: #28a745;
            color: white;
        }}
        
        .status.failed {{
            background: #dc3545;
            color: white;
        }}
        
        .result-details {{
            font-size: 14px;
            color: #666;
        }}
        
        .result-details p {{
            margin-bottom: 5px;
        }}
        
        .prompt-text {{
            background: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 13px;
            color: #495057;
            max-height: 80px;
            overflow-y: auto;
        }}
        
        .stats {{
            display: flex;
            gap: 15px;
            margin-top: 10px;
            font-size: 12px;
            color: #6c757d;
        }}
        
        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: 1fr;
            }}
            
            .config-grid {{
                grid-template-columns: 1fr;
            }}
            
            .results-grid {{
                grid-template-columns: 1fr;
            }}
            
            .filters {{
                flex-direction: column;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Lumina2模型测试结果报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-section">
            <h2>📊 测试概览</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>{test_summary['total_tests']}</h3>
                    <p>总测试数</p>
                </div>
                <div class="summary-card">
                    <h3>{test_summary['successful_tests']}</h3>
                    <p>成功测试</p>
                </div>
                <div class="summary-card">
                    <h3>{test_summary['failed_tests']}</h3>
                    <p>失败测试</p>
                </div>
                <div class="summary-card">
                    <h3>{test_summary['success_rate']:.1f}%</h3>
                    <p>成功率</p>
                </div>
            </div>
        </div>
        
        <div class="config-section">
            <h2>⚙️ 测试配置</h2>
            <div class="config-grid">
                <div class="config-item">
                    <h4>模型</h4>
                    <p>{len(test_summary['test_config']['models'])} 个模型</p>
                    <ul>
                        {''.join([f'<li>{os.path.basename(model)}</li>' for model in test_summary['test_config']['models']])}
                    </ul>
                </div>
                <div class="config-item">
                    <h4>提示词</h4>
                    <p>{len(test_summary['test_config']['prompts'])} 个提示词</p>
                </div>
                <div class="config-item">
                    <h4>步数</h4>
                    <p>{len(test_summary['test_config']['steps_list'])} 种步数: {', '.join(map(str, test_summary['test_config']['steps_list']))}</p>
                </div>
                <div class="config-item">
                    <h4>CFG值</h4>
                    <p>{len(test_summary['test_config']['cfg_list'])} 种CFG: {', '.join(map(str, test_summary['test_config']['cfg_list']))}</p>
                </div>
                <div class="config-item">
                    <h4>图像尺寸</h4>
                    <p>{test_summary['test_config']['image_width']} × {test_summary['test_config']['image_height']}</p>
                </div>
                <div class="config-item">
                    <h4>随机种子</h4>
                    <p>{test_summary['test_config']['seed']}</p>
                </div>
            </div>
        </div>
        
        <div class="results-section">
            <h2>🖼️ 测试结果</h2>
            
            <div class="filters">
                <div class="filter-group">
                    <label>状态:</label>
                    <select id="statusFilter">
                        <option value="">全部</option>
                        <option value="success">成功</option>
                        <option value="failed">失败</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>模型:</label>
                    <select id="modelFilter">
                        <option value="">全部</option>
                        {''.join([f'<option value="{os.path.basename(model)}">{os.path.basename(model)}</option>' for model in test_summary['test_config']['models']])}
                    </select>
                </div>
                <div class="filter-group">
                    <label>步数:</label>
                    <select id="stepsFilter">
                        <option value="">全部</option>
                        {''.join([f'<option value="{steps}">{steps}</option>' for steps in test_summary['test_config']['steps_list']])}
                    </select>
                </div>
                <div class="filter-group">
                    <label>CFG:</label>
                    <select id="cfgFilter">
                        <option value="">全部</option>
                        {''.join([f'<option value="{cfg}">{cfg}</option>' for cfg in test_summary['test_config']['cfg_list']])}
                    </select>
                </div>
            </div>
            
            <div class="results-grid" id="resultsGrid">
"""
    
    # 添加每个测试结果
    for result in test_results:
        # 使用绝对路径
        image_path = os.path.join(test_summary['test_config']['output_dir'], result['output_image']) if result['output_image'] else None
        image_html = f'<img src="file://{image_path}" alt="测试图像 {result["test_id"]}">' if image_path and result['status'] == 'success' else '<div>无图像</div>'
        
        html_content += f"""
                <div class="result-card" data-status="{result['status']}" data-model="{result['model_name']}" data-steps="{result['steps']}" data-cfg="{result['cfg']}">
                    <div class="result-image">
                        {image_html}
                    </div>
                    <div class="result-info">
                        <div class="result-header">
                            <span class="test-id">{result['test_id']}</span>
                            <span class="status {result['status']}">{result['status'].upper()}</span>
                        </div>
                        <div class="result-details">
                            <p><strong>模型:</strong> {result['model_name']}</p>
                            <p><strong>步数:</strong> {result['steps']}</p>
                            <p><strong>CFG:</strong> {result['cfg']}</p>
                            <p><strong>生成时间:</strong> {result['generation_time']:.2f}秒</p>
                            <p><strong>时间戳:</strong> {result['timestamp']}</p>
                            {f'<p><strong>错误:</strong> {result["error"]}</p>' if result['status'] == 'failed' else ''}
                        </div>
                        <div class="prompt-text">
                            <strong>提示词:</strong><br>
                            {result['prompt'][:100]}{'...' if len(result['prompt']) > 100 else ''}
                        </div>
                        <div class="stats">
                            <span>尺寸: {result['image_width']}×{result['image_height']}</span>
                            <span>种子: {result['seed']}</span>
                        </div>
                    </div>
                </div>
"""
    
    # 添加JavaScript过滤功能
    html_content += """
            </div>
        </div>
    </div>
    
    <script>
        // 过滤功能
        function filterResults() {
            const statusFilter = document.getElementById('statusFilter').value;
            const modelFilter = document.getElementById('modelFilter').value;
            const stepsFilter = document.getElementById('stepsFilter').value;
            const cfgFilter = document.getElementById('cfgFilter').value;
            
            const cards = document.querySelectorAll('.result-card');
            
            cards.forEach(card => {
                const status = card.dataset.status;
                const model = card.dataset.model;
                const steps = card.dataset.steps;
                const cfg = card.dataset.cfg;
                
                const statusMatch = !statusFilter || status === statusFilter;
                const modelMatch = !modelFilter || model === modelFilter;
                const stepsMatch = !stepsFilter || steps === stepsFilter;
                const cfgMatch = !cfgFilter || cfg === cfgFilter;
                
                if (statusMatch && modelMatch && stepsMatch && cfgMatch) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
        
        // 添加事件监听器
        document.getElementById('statusFilter').addEventListener('change', filterResults);
        document.getElementById('modelFilter').addEventListener('change', filterResults);
        document.getElementById('stepsFilter').addEventListener('change', filterResults);
        document.getElementById('cfgFilter').addEventListener('change', filterResults);
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {
            console.log('测试结果页面加载完成');
        });
    </script>
</body>
</html>
"""
    
    # 保存HTML文件
    with open(output_html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML报告已生成: {output_html_path}")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print("使用方法: python visualize_results.py <json_file_path>")
        print("示例: python visualize_results.py ./output/simple_test/test_results.json")
        return
    
    json_file_path = sys.argv[1]
    
    if not os.path.exists(json_file_path):
        print(f"错误: JSON文件不存在: {json_file_path}")
        return
    
        # 读取JSON数据来获取输出目录
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 生成HTML文件路径 - 保存到与图像相同的目录
    output_dir = data['test_summary']['test_config']['output_dir']
    output_html_path = os.path.join(output_dir, 'test_results_report.html')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_html_report(json_file_path, output_html_path)
        print(f"✅ HTML报告生成成功!")
        print(f"📁 报告路径: {output_html_path}")
        print(f"🌐 请在浏览器中打开查看")
    except Exception as e:
        print(f"❌ 生成HTML报告失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 