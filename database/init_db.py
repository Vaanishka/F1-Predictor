"""
Database Initialization - 25 Feature System

Usage:
    python init_db.py              # Create database
    python init_db.py --reset      # Delete and recreate
    python init_db.py --inspect    # View contents
"""

import sqlite3
from pathlib import Path
from datetime import datetime
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / 'database'
DB_FILE = DB_DIR / 'f1_predictions.db'
SCHEMA_FILE = DB_DIR / 'schema.sql'

DB_DIR.mkdir(parents=True, exist_ok=True)


def reset_database():
    """Delete existing database"""
    if DB_FILE.exists():
        DB_FILE.unlink()
        print(f"✅ Deleted existing database")
    return True


def create_database():
    """Create database from schema"""
    
    print("\n" + "="*70)
    print("🗄️  F1 PREDICTION DATABASE - 25 FEATURES (UNBIASED)")
    print("="*70)
    
    if not SCHEMA_FILE.exists():
        print(f"\n❌ Schema file not found: {SCHEMA_FILE}")
        return False
    
    with open(SCHEMA_FILE, 'r') as f:
        schema_sql = f.read()
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.executescript(schema_sql)
        conn.commit()
        
        print(f"\n✅ Database created: {DB_FILE}")
        
        # Verify structure
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = cursor.fetchall()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name")
        views = cursor.fetchall()
        
        print(f"\n📊 Database Structure:")
        print(f"   Tables: {len(tables)}")
        for table in tables:
            print(f"      • {table[0]}")
        
        print(f"\n   Views: {len(views)}")
        for view in views:
            print(f"      • {view[0]}")
        
        # Check model
        cursor.execute("SELECT COUNT(*) FROM model_metadata WHERE is_active = 1")
        active_models = cursor.fetchone()[0]
        
        if active_models > 0:
            cursor.execute("""
                SELECT model_version, test_mae, test_r2, notes 
                FROM model_metadata 
                WHERE is_active = 1
            """)
            model_info = cursor.fetchone()
            
            print(f"\n📈 Active Model:")
            print(f"   Version: {model_info[0]}")
            print(f"   Test MAE: {model_info[1]:.2f}")
            print(f"   Test R²: {model_info[2]:.2f}")
            print(f"   Notes: {model_info[3][:60]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    finally:
        conn.close()


def inspect_database():
    """View database contents"""
    
    print("\n" + "="*70)
    print("🔍 DATABASE INSPECTION")
    print("="*70)
    
    if not DB_FILE.exists():
        print(f"\n❌ Database not found: {DB_FILE}")
        print("   Run: python init_db.py first")
        return False
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Table counts
        print("\n📊 Table Row Counts:")
        
        tables = ['races', 'race_features', 'predictions', 'actual_results', 
                 'model_performance', 'model_metadata']
        
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   {table:<20s} {count:>6,} rows")
        
        # Active model
        cursor.execute("""
            SELECT model_version, test_mae, test_r2
            FROM model_metadata
            WHERE is_active = 1
        """)
        model_info = cursor.fetchone()
        
        if model_info:
            print(f"\n📈 Active Model:")
            print(f"   Version: {model_info[0]}")
            print(f"   MAE: {model_info[1]:.2f} | R²: {model_info[2]:.2f}")
        
        # Recent races
        cursor.execute("""
            SELECT year, race_name, status
            FROM races
            ORDER BY year DESC, race_name
            LIMIT 10
        """)
        races = cursor.fetchall()
        
        if races:
            print(f"\n🏁 Races:")
            for race in races:
                print(f"   {race[0]} - {race[1]:<25s} [{race[2]}]")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
    
    finally:
        conn.close()


def main():
    """Main initialization"""
    
    import argparse
    parser = argparse.ArgumentParser(description='Initialize F1 Database')
    parser.add_argument('--reset', action='store_true', help='Delete and recreate')
    parser.add_argument('--inspect', action='store_true', help='Inspect contents')
    
    args = parser.parse_args()
    
    if args.inspect:
        inspect_database()
        return
    
    if args.reset:
        if DB_FILE.exists():
            response = input(f"\n⚠️  Delete {DB_FILE}? (yes/no): ")
            if response.lower() == 'yes':
                reset_database()
            else:
                print("Cancelled")
                return
    
    if DB_FILE.exists() and not args.reset:
        print(f"\n⚠️  Database already exists: {DB_FILE}")
        print("   Use --reset to recreate, or --inspect to view")
        return
    
    success = create_database()
    
    if success:
        print("\n" + "="*70)
        print("✅ DATABASE READY!")
        print("="*70)
        print(f"\n📁 Location: {DB_FILE}")
        print(f"\n🎯 Next Steps:")
        print(f"   1. python init_db.py --inspect")
        print(f"   2. cd ../scripts")
        print(f"   3. python extract_race.py --race 'Bahrain Grand Prix' --year 2024")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()