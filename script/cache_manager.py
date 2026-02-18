import os
import json
import shutil
from pathlib import Path
import logging
import fastf1

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1CacheManager:
    def __init__(self, cache_dir='./f1_cache'):
        self.cache_dir = Path(cache_dir)
        
    def analyze_cache(self):
        """Analyze cache directory and report status"""
        if not self.cache_dir.exists():
            logger.info(f"Cache directory does not exist: {self.cache_dir}")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Cache Analysis: {self.cache_dir}")
        logger.info(f"{'='*60}")
        
        # Get all files
        cache_files = list(self.cache_dir.rglob('*'))
        
        total_size = sum(f.stat().st_size for f in cache_files if f.is_file())
        
        logger.info(f"Total files: {len([f for f in cache_files if f.is_file()])}")
        logger.info(f"Total directories: {len([f for f in cache_files if f.is_dir()])}")
        logger.info(f"Total size: {total_size / (1024**2):.2f} MB")
        
        # Check for corrupted files
        corrupted = self.check_corrupted_files()
        if corrupted:
            logger.warning(f"Found {len(corrupted)} potentially corrupted files")
            for f in corrupted[:10]:  # Show first 10
                logger.warning(f"  - {f}")
        
        return {
            'total_files': len([f for f in cache_files if f.is_file()]),
            'total_size_mb': total_size / (1024**2),
            'corrupted_files': corrupted
        }
    
    def check_corrupted_files(self):
        """Check for potentially corrupted cache files"""
        corrupted = []
        
        if not self.cache_dir.exists():
            return corrupted
        
        for cache_file in self.cache_dir.rglob('*.ff1pkl'):
            try:
                # Check if file is empty or very small
                if cache_file.stat().st_size < 100:
                    corrupted.append(str(cache_file))
            except Exception as e:
                logger.error(f"Error checking {cache_file}: {e}")
                corrupted.append(str(cache_file))
        
        return corrupted
    
    def clear_cache(self, confirm=True):
        """Clear entire cache directory"""
        if not self.cache_dir.exists():
            logger.info("Cache directory does not exist")
            return
        
        if confirm:
            response = input(f"Are you sure you want to delete {self.cache_dir}? (yes/no): ")
            if response.lower() != 'yes':
                logger.info("Cache clear cancelled")
                return
        
        try:
            shutil.rmtree(self.cache_dir)
            logger.info(f"✅ Cache cleared: {self.cache_dir}")
            self.cache_dir.mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def clear_corrupted_only(self):
        """Remove only corrupted files from cache"""
        corrupted = self.check_corrupted_files()
        
        if not corrupted:
            logger.info("No corrupted files found")
            return
        
        logger.info(f"Removing {len(corrupted)} corrupted files...")
        
        for file_path in corrupted:
            try:
                os.remove(file_path)
                logger.info(f"Removed: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove {file_path}: {e}")
        
        logger.info("✅ Corrupted files removed")
    
    def verify_session_data(self, year, gp_name, session_type):
        """Verify if a specific session's data is valid"""
        logger.info(f"\nVerifying: {year} {gp_name} {session_type}")
        
        try:
            fastf1.Cache.enable_cache(str(self.cache_dir))
            session = fastf1.get_session(year, gp_name, session_type)
            session.load(telemetry=True)
            
            # Check laps
            if not hasattr(session, 'laps') or session.laps.empty:
                logger.error("❌ No lap data")
                return False
            
            logger.info(f"✅ Found {len(session.laps)} laps")
            
            # Check telemetry for first valid lap
            valid_laps = session.laps[session.laps['LapTime'].notna()]
            if valid_laps.empty:
                logger.error("❌ No valid laps with lap times")
                return False
            
            first_lap = valid_laps.iloc[0]
            tel = first_lap.get_telemetry()
            
            if tel is None or tel.empty:
                logger.error("❌ No telemetry data")
                return False
            
            logger.info(f"✅ Telemetry columns: {list(tel.columns)}")
            
            # Check for required channels
            required = ['Speed', 'Throttle', 'Brake']
            missing = [col for col in required if col not in tel.columns]
            
            if missing:
                logger.warning(f"⚠️  Missing channels: {missing}")
            else:
                logger.info("✅ All required channels present")
            
            # Check for empty data
            for col in tel.columns:
                non_null = tel[col].notna().sum()
                logger.info(f"  {col}: {non_null}/{len(tel)} non-null values")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Verification failed: {str(e)}")
            return False
    
    def rebuild_cache_for_session(self, year, gp_name, session_type):
        """Rebuild cache for a specific session"""
        logger.info(f"Rebuilding cache for {year} {gp_name} {session_type}")
        
        # Find and remove existing cache files for this session
        session_pattern = f"{year}_{gp_name}_{session_type}"
        
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file() and session_pattern.lower() in cache_file.name.lower():
                try:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache: {cache_file.name}")
                except Exception as e:
                    logger.error(f"Failed to remove {cache_file}: {e}")
        
        # Reload the session
        try:
            fastf1.Cache.enable_cache(str(self.cache_dir))
            session = fastf1.get_session(year, gp_name, session_type)
            session.load(telemetry=True, weather=True, messages=True)
            logger.info(f"✅ Cache rebuilt for {year} {gp_name} {session_type}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to rebuild cache: {str(e)}")
            return False


def diagnose_telemetry_issue():
    """Diagnose the telemetry issue from the error logs"""
    logger.info("\n" + "="*60)
    logger.info("TELEMETRY ISSUE DIAGNOSIS")
    logger.info("="*60)
    
    logger.info("""
    Based on the error logs, here's what's happening:
    
    1. ERROR: "Telemetry does not contain required channels 'Time' and 'Speed'"
       - This affects ALL 20 drivers in the session
       - This is a systematic issue, not random corruption
    
    2. POSSIBLE CAUSES:
       a) Cache files are corrupted (most likely)
       b) FastF1 API has changed and data structure is different
       c) The session data itself is incomplete on the F1 servers
       d) Network issue during initial download
    
    3. SOLUTION STEPS:
       a) Clear the entire cache (force fresh download)
       b) Remove the 'results' parameter from Session.load()
       c) Add better error handling for missing telemetry
       d) Verify data immediately after loading
    
    4. PREVENTION:
       - Always verify telemetry after loading
       - Use try-except blocks around telemetry access
       - Implement fallback to lap-level data if telemetry fails
       - Cache verification before processing
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 Cache Manager')
    parser.add_argument('--analyze', action='store_true', help='Analyze cache')
    parser.add_argument('--clear', action='store_true', help='Clear entire cache')
    parser.add_argument('--clear-corrupted', action='store_true', help='Clear only corrupted files')
    parser.add_argument('--verify', nargs=3, metavar=('YEAR', 'GP', 'SESSION'), 
                       help='Verify specific session (e.g., 2024 "Italian Grand Prix" Q)')
    parser.add_argument('--rebuild', nargs=3, metavar=('YEAR', 'GP', 'SESSION'),
                       help='Rebuild cache for specific session')
    parser.add_argument('--diagnose', action='store_true', help='Run diagnosis')
    
    args = parser.parse_args()
    
    manager = F1CacheManager()
    
    if args.analyze:
        manager.analyze_cache()
    elif args.clear:
        manager.clear_cache()
    elif args.clear_corrupted:
        manager.clear_corrupted_only()
    elif args.verify:
        year, gp, session = args.verify
        manager.verify_session_data(int(year), gp, session)
    elif args.rebuild:
        year, gp, session = args.rebuild
        manager.rebuild_cache_for_session(int(year), gp, session)
    elif args.diagnose:
        diagnose_telemetry_issue()
    else:
        print("No action specified. Use --help for options")
        
        # Run default analysis
        manager.analyze_cache()
        diagnose_telemetry_issue()