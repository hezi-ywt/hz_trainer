#!/usr/bin/env python3
"""
快速启动HTTP服务器访问模型测试结果可视化页面
"""

import os
import sys
import argparse
import http.server
import socketserver
from pathlib import Path

def start_http_server(html_file_path, port, host="localhost"):
    """启动HTTP服务器"""
    import signal
    import sys
    
    # 检查HTML文件是否存在
    if not os.path.exists(html_file_path):
        print(f"❌ HTML文件不存在: {html_file_path}")
        return False
    
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
        print(f"📡 服务器地址: http://{host}:{port}")
        print(f"📄 可视化页面: http://{host}:{port}/{html_filename}")
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

def find_html_file(directory):
    """在指定目录中查找HTML文件"""
    html_files = list(Path(directory).glob("*.html"))
    if html_files:
        return str(html_files[0])
    return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='启动HTTP服务器访问模型测试结果可视化页面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python serve_visualization.py
  python serve_visualization.py -p 8080
  python serve_visualization.py -f visualization.html -p 8080
  python serve_visualization.py -d /path/to/html/directory -p 8080
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        help='HTML文件路径 (默认: 自动查找当前目录下的HTML文件)'
    )
    
    parser.add_argument(
        '-d', '--directory',
        help='HTML文件所在目录 (默认: 当前目录)'
    )
    
    parser.add_argument(
        '-p', '--port',
        type=int,
        default=8080,
        help='服务器端口 (默认: 8080)'
    )
    
    parser.add_argument(
        '--host',
        default='localhost',
        help='服务器主机地址 (默认: localhost)'
    )
    
    args = parser.parse_args()
    
    # 确定HTML文件路径
    html_file_path = None
    
    if args.file:
        # 用户指定了HTML文件
        html_file_path = args.file
        if not os.path.isabs(html_file_path):
            html_file_path = os.path.abspath(html_file_path)
    else:
        # 自动查找HTML文件
        search_dir = args.directory if args.directory else os.getcwd()
        print(f"🔍 在目录中查找HTML文件: {search_dir}")
        
        html_file_path = find_html_file(search_dir)
        
        if not html_file_path:
            print(f"❌ 在目录 {search_dir} 中未找到HTML文件")
            print("💡 请使用 -f 参数指定HTML文件路径")
            return
    
    print(f"📄 找到HTML文件: {html_file_path}")
    
    # 启动HTTP服务器
    start_http_server(html_file_path, args.port, args.host)

if __name__ == "__main__":
    main()
