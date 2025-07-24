"""
Startup script for Student Performance Prediction Web Application
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = {
        'flask': 'flask',
        'pandas': 'pandas', 
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # Note: package name vs import name difference
        'plotly': 'plotly',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name} found")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} missing")
    
    if missing_packages:
        print("\n❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall missing packages with:")
        print(f"   pip3 install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies found")
    return True

def check_data_files():
    """Check if data files are available"""
    data_dir = project_root / 'data'
    if not data_dir.exists():
        print("❌ Data directory not found. Creating...")
        data_dir.mkdir(exist_ok=True)
        print("📁 Please add your CSV datasets to the 'data/' directory")
        return False
    
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print("⚠️  No CSV files found in data/ directory")
        print("📁 Please add your datasets to continue")
        return False
    
    print(f"✅ Found {len(csv_files)} dataset(s):")
    for file in csv_files:
        print(f"   - {file.name}")
    
    return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'models/trained',
        'reports/interpretability',
        'logs',
        'web/static/css',
        'web/static/js',
        'web/static/images'
    ]
    
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("📁 Created necessary directories")

def run_flask_app(host='localhost', port=5000, debug=False):
    """Run the Flask application"""
    try:
        from web.app import app
        
        print(f"\n🚀 Starting Student Performance Prediction Web App")
        print(f"📊 Dashboard: http://{host}:{port}")
        print(f"🔮 Upload Class Data: http://{host}:{port}/predict")
        print(f"✅ Model Validation: http://{host}:{port}/validate")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60)
        
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting Flask app: {e}")
        return False
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Student Performance Prediction Web Application'
    )
    parser.add_argument(
        '--host', 
        default='localhost',
        help='Host to bind to (default: localhost)'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000,
        help='Port to bind to (default: 5000)'
    )
    parser.add_argument(
        '--debug', 
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--skip-checks', 
        action='store_true',
        help='Skip dependency and data checks'
    )
    
    args = parser.parse_args()
    
    print("🎓 Student Performance Prediction Web Application")
    print("=" * 60)
    
    if not args.skip_checks:
        # Check dependencies
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        print("✅ All dependencies found")
        
        # Check data files
        print("\n🔍 Checking data files...")
        data_available = check_data_files()
        if not data_available:
            print("⚠️  You can still run the app, but training won't work without data")
    
    # Create directories
    print("\n📁 Setting up directories...")
    create_directories()
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development' if args.debug else 'production'
    
    # Run the application
    success = run_flask_app(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main() 