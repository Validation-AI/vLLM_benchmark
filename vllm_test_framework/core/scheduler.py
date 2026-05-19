# core/scheduler.py
import logging
import time
import requests
from typing import Dict, List, Optional, Callable
from .models import TestCase, TestResult
from .server_manager import ServerManager
from .client_tester import ClientTester
from .result_collector import ResultCollector
from .db_updater import DBUpdater

logger = logging.getLogger(__name__)

class TestScheduler:
    """
    测试调度器，负责协调 Server 和 Client 的执行
    """
    
    def __init__(self, 
                 server_manager: ServerManager,
                 client_tester: ClientTester,
                 result_collector: ResultCollector,
                 db_updater: DBUpdater):
        self.server_manager = server_manager
        self.client_tester = client_tester
        self.result_collector = result_collector
        self.db_updater = db_updater
        
        # 状态跟踪
        self.current_category: Optional[str] = None
        self.is_server_running: bool = False
        
        # 回调函数
        self.on_category_start: Optional[Callable] = None
        self.on_category_complete: Optional[Callable] = None
        self.on_test_complete: Optional[Callable] = None

    def run_test_category(self, 
                         category: str, 
                         cases: List[TestCase],
                         server_start_delay: int = 15) -> List[TestResult]:
        """
        执行一个测试类别的完整流程
        """
        self.current_category = category
        
        # 回调：类别开始
        if self.on_category_start:
            self.on_category_start(category, cases)
            
        logger.info(f"Starting test category: {category} with {len(cases)} cases")
        
        try:
            # 启动 Server
            self._start_server(server_start_delay)
            
            # 执行测试
            results = self._execute_tests(cases)
            
            # 收集结果
            self.result_collector.add_results(results)
            
            # 更新数据库
            self._update_database(results)
            
            # 回调：测试完成
            if self.on_test_complete:
                self.on_test_complete(category, results)
                
            return results
            
        except Exception as e:
            logger.error(f"Error during test category {category}: {e}")
            raise
            
        finally:
            # 确保停止 Server
            self._stop_server()
            
            # 回调：类别完成
            if self.on_category_complete:
                self.on_category_complete(category)

    def run_all_categories(self, 
                          grouped_cases: Dict[str, List[TestCase]],
                          server_start_delay: int = 15) -> Dict[str, List[TestResult]]:
        """
        执行所有测试类别
        """
        all_results = {}
        
        for category, cases in grouped_cases.items():
            try:
                results = self.run_test_category(category, cases, server_start_delay)
                all_results[category] = results
                
            except Exception as e:
                logger.error(f"Failed to run category {category}: {e}")
                # 可以选择继续执行其他类别或终止
                continue
                
        return all_results

    def _start_server(self, start_delay: int):
        """启动 Server"""
        if self.is_server_running:
            logger.warning("Server is already running, skipping start")
            return
            
        logger.info("Starting vLLM server...")
        self.server_manager.start_server(start_delay)
        self.is_server_running = True
        
        # 等待服务器完全就绪
        self._wait_for_server_ready()

    def _stop_server(self):
        """停止 Server"""
        if not self.is_server_running:
            return
            
        logger.info("Stopping vLLM server...")
        self.server_manager.stop_server()
        self.is_server_running = False
        self.current_category = None

    def _wait_for_server_ready(self, max_attempts: int = 10, delay: int = 3):
        """
        等待 Server 完全就绪
        """
        logger.info("Waiting for server to be ready...")
        
        for attempt in range(max_attempts):
            try:
                # 简单的健康检查
                health_url = f"http://{self.client_tester.args.host}:{self.client_tester.args.port}/health"
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    return
                    
            except requests.exceptions.RequestException:
                pass
                
            if attempt < max_attempts - 1:
                logger.debug(f"Server not ready yet, retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.warning("Server health check failed, but continuing...")

    def _execute_tests(self, cases: List[TestCase]) -> List[TestResult]:
        """执行测试用例"""
        logger.info(f"Executing {len(cases)} test cases...")
        
        # 可以根据需要实现不同的执行策略
        return self._execute_sequential(cases)
        
    def _execute_sequential(self, cases: List[TestCase]) -> List[TestResult]:
        """顺序执行测试用例"""
        return self.client_tester.test_batch(cases)
    
    def _execute_parallel(self, cases: List[TestCase], max_workers: int = 4) -> List[TestResult]:
        """并行执行测试用例（可选实现）"""
        # 这里可以实现并行测试逻辑
        # 由于 vLLM 本身支持并发，通常顺序执行即可
        return self.client_tester.test_batch(cases)

    def _update_database(self, results: List[TestResult]):
        """更新数据库"""
        logger.info(f"Updating {len(results)} results to database...")
        self.db_updater.update_results(results)

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "current_category": self.current_category,
            "is_server_running": self.is_server_running,
            "total_results": len(self.result_collector.all_results),
            "summary": self.result_collector.get_summary()
        }

    def register_callbacks(self,
                          on_category_start: Optional[Callable] = None,
                          on_category_complete: Optional[Callable] = None,
                          on_test_complete: Optional[Callable] = None):
        """注册回调函数"""
        self.on_category_start = on_category_start
        self.on_category_complete = on_category_complete
        self.on_test_complete = on_test_complete