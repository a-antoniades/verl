import logging
import os
import shutil
from moatless.runtime.testbed import TestbedEnvironment
from moatless.benchmark.swebench import create_repository, create_index
from moatless.benchmark.evaluation import load_instances

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_testbed(instance_index=1):  # Default to second instance (index 1)
    # Load instances from the verified dataset
    instances = load_instances("verified")
    
    # Print available instances
    logger.info("Available instances:")
    for i, inst in enumerate(instances[:5]):  # Show first 5 instances
        logger.info(f"{i}: {inst['instance_id']}")
    
    # Get the specified instance
    instance = instances[instance_index]
    
    # Setup paths
    repo_base_dir = "/tmp/test_repos"
    log_dir = "/tmp/test_logs"
    os.makedirs(log_dir, exist_ok=True)

    try:
        # Create repository
        logger.info(f"Creating repository for instance {instance['instance_id']}...")
        repository = create_repository(instance, repo_base_dir=repo_base_dir)
        
        # Create code index
        logger.info("Creating code index...")
        code_index = create_index(instance, repository=repository)

        # Initialize testbed
        logger.info("Initializing testbed environment...")
        testbed = TestbedEnvironment(
            repository=repository,
            instance=instance,
            log_dir=log_dir,
            dataset_name="princeton-nlp/SWE-bench_Verified",
            timeout=2000
        )

        # Get the golden patch from the instance
        test_patch = instance.get("golden_patch", "")
        if not test_patch:
            logger.error("No golden patch found in instance")
            return False

        # Print test environment info
        logger.info(f"Repository directory: {repository.repo_dir}")
        logger.info(f"Test environment initialized with dataset: {testbed.dataset_name}")

        # Evaluate the patch
        logger.info("Testing patch evaluation...")
        result = testbed.evaluate(patch=test_patch)
        
        # Print results
        logger.info(f"Evaluation completed. Result: {result}")
        if result:
            logger.info(f"Resolved: {result.resolved}")
            # Get test results from tests_status
            if result.tests_status:
                logger.info(f"Test status: {result.tests_status.status}")
                if result.tests_status.fail_to_pass:
                    logger.info(f"Failed tests: {result.tests_status.fail_to_pass.failure}")
                    logger.info(f"Passed tests: {result.tests_status.fail_to_pass.success}")
                if hasattr(result.tests_status, 'error_message') and result.tests_status.error_message:
                    logger.info(f"Error message: {result.tests_status.error_message}")

        return result is not None

    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        # Cleanup
        logger.info("Cleaning up test directories...")
        if os.path.exists(repo_base_dir):
            shutil.rmtree(repo_base_dir, ignore_errors=True)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir, ignore_errors=True)

if __name__ == "__main__":
    # Try with second instance (index 1)
    success = test_testbed(instance_index=1)
    print(f"\nTestbed test {'PASSED' if success else 'FAILED'}")