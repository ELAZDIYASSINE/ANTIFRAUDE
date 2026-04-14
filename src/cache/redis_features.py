#!/usr/bin/env python3
"""
Redis Cache for Real-time Features
Features: Feature caching, real-time lookups, performance optimization
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("redis not available - caching disabled")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisFeatureCache:
    """Redis-based feature cache for real-time fraud detection"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, 
                 db: int = 0, password: Optional[str] = None):
        """Initialize Redis feature cache"""
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.redis_client = None
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
        
    def connect(self) -> bool:
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - caching disabled")
            return False
        
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            return False
    
    def cache_transaction_features(self, transaction_id: str, features: Dict, 
                                  ttl: int = 3600) -> bool:
        """Cache transaction features"""
        if not self.redis_client:
            return False
        
        try:
            key = f"transaction:{transaction_id}"
            value = json.dumps(features)
            
            self.redis_client.setex(key, ttl, value)
            self.cache_stats['sets'] += 1
            
            logger.debug(f"✅ Cached features for transaction {transaction_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching features: {e}")
            return False
    
    def get_cached_features(self, transaction_id: str) -> Optional[Dict]:
        """Get cached transaction features"""
        if not self.redis_client:
            return None
        
        try:
            key = f"transaction:{transaction_id}"
            value = self.redis_client.get(key)
            
            if value:
                self.cache_stats['hits'] += 1
                features = json.loads(value)
                logger.debug(f"✅ Cache hit for transaction {transaction_id}")
                return features
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"❌ Cache miss for transaction {transaction_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached features: {e}")
            return None
    
    def cache_account_features(self, account_id: str, features: Dict,
                             ttl: int = 1800) -> bool:
        """Cache account-level features"""
        if not self.redis_client:
            return False
        
        try:
            key = f"account:{account_id}"
            value = json.dumps(features)
            
            self.redis_client.setex(key, ttl, value)
            self.cache_stats['sets'] += 1
            
            logger.debug(f"✅ Cached features for account {account_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching account features: {e}")
            return False
    
    def get_cached_account_features(self, account_id: str) -> Optional[Dict]:
        """Get cached account features"""
        if not self.redis_client:
            return None
        
        try:
            key = f"account:{account_id}"
            value = self.redis_client.get(key)
            
            if value:
                self.cache_stats['hits'] += 1
                features = json.loads(value)
                logger.debug(f"✅ Cache hit for account {account_id}")
                return features
            else:
                self.cache_stats['misses'] += 1
                logger.debug(f"❌ Cache miss for account {account_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached account features: {e}")
            return None
    
    def cache_risk_score(self, entity_id: str, risk_score: float, 
                        risk_level: str, ttl: int = 300) -> bool:
        """Cache risk score for entity"""
        if not self.redis_client:
            return False
        
        try:
            key = f"risk:{entity_id}"
            value = json.dumps({
                'risk_score': risk_score,
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat()
            })
            
            self.redis_client.setex(key, ttl, value)
            self.cache_stats['sets'] += 1
            
            logger.debug(f"✅ Cached risk score for {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error caching risk score: {e}")
            return False
    
    def get_cached_risk_score(self, entity_id: str) -> Optional[Dict]:
        """Get cached risk score"""
        if not self.redis_client:
            return None
        
        try:
            key = f"risk:{entity_id}"
            value = self.redis_client.get(key)
            
            if value:
                self.cache_stats['hits'] += 1
                return json.loads(value)
            else:
                self.cache_stats['misses'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error getting cached risk score: {e}")
            return None
    
    def invalidate_cache(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.cache_stats['deletes'] += deleted
                logger.info(f"✅ Invalidated {deleted} cache entries matching {pattern}")
                return deleted
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate_percent': hit_rate,
            'sets': self.cache_stats['sets'],
            'deletes': self.cache_stats['deletes'],
            'total_requests': total_requests
        }
    
    def flush_cache(self) -> bool:
        """Flush all cache entries"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.flushdb()
            logger.info("✅ Cache flushed")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {e}")
            return False
    
    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            self.redis_client.close()
            logger.info("✅ Redis connection closed")


class RealTimeFeatureService:
    """Real-time feature service with Redis caching"""
    
    def __init__(self, redis_cache: RedisFeatureCache):
        """Initialize real-time feature service"""
        self.cache = redis_cache
        
    def get_transaction_features(self, transaction: Dict) -> Dict:
        """Get features for transaction with caching"""
        transaction_id = transaction.get('nameOrig', 'unknown')
        
        # Try cache first
        cached_features = self.cache.get_cached_features(transaction_id)
        if cached_features:
            return cached_features
        
        # Calculate features (mock - would normally compute)
        features = {
            'amount': transaction.get('amount', 0.0),
            'oldbalanceOrg': transaction.get('oldbalanceOrg', 0.0),
            'newbalanceOrig': transaction.get('newbalanceOrig', 0.0),
            'oldbalanceDest': transaction.get('oldbalanceDest', 0.0),
            'newbalanceDest': transaction.get('newbalanceDest', 0.0),
            'type': transaction.get('type', 'UNKNOWN'),
            'amount_ratio': transaction.get('amount', 0.0) / max(transaction.get('oldbalanceOrg', 1), 1),
            'balance_change_orig': transaction.get('oldbalanceOrg', 0.0) - transaction.get('newbalanceOrig', 0.0),
            'balance_change_dest': transaction.get('newbalanceDest', 0.0) - transaction.get('oldbalanceDest', 0.0),
            'is_large_amount': 1 if transaction.get('amount', 0.0) > 100000 else 0
        }
        
        # Cache for future use
        self.cache.cache_transaction_features(transaction_id, features)
        
        return features
    
    def get_account_risk_history(self, account_id: str, limit: int = 10) -> List[Dict]:
        """Get risk history for account"""
        # Mock implementation - would normally query database
        return [
            {
                'transaction_id': f'TX{i}',
                'risk_score': 0.5 + (i * 0.05),
                'timestamp': datetime.now().isoformat()
            }
            for i in range(limit)
        ]


def main():
    """Main execution for Redis feature cache"""
    try:
        cache = RedisFeatureCache()
        
        if not cache.connect():
            logger.warning("Redis not available - using mock cache")
            return 0
        
        # Test caching
        transaction_id = "TEST_TX_001"
        features = {
            'amount': 150000.0,
            'oldbalanceOrg': 200000.0,
            'newbalanceOrig': 50000.0,
            'risk_score': 0.85
        }
        
        # Cache features
        cache.cache_transaction_features(transaction_id, features)
        
        # Retrieve cached features
        cached = cache.get_cached_features(transaction_id)
        
        # Get stats
        stats = cache.get_cache_stats()
        
        print("\n" + "=" * 60)
        print("📊 REDIS CACHE TEST")
        print("=" * 60)
        print(f"Cached features: {cached is not None}")
        print(f"Cache stats: {stats}")
        print("=" * 60)
        
        cache.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Redis cache test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
