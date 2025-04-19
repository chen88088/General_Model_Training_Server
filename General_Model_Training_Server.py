from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
import logging
import requests
import subprocess
from pydantic import BaseModel
from typing import List
from pathlib import Path
import os
import shutil
import git
from DVCManager import DVCManager
from DVCWorker import DVCWorker
from typing import Dict
from LoggerManager import LoggerManager
from kubernetes import client, config
from DVCManager import DVCManager
from DVCWorker import DVCWorker
from typing import Dict
from LoggerManager import LoggerManager
from DagManager import DagManager
from fastapi.responses import JSONResponse
import yaml  # 解析 DVC 檔案
import shutil
import re  # 引入正則表達式套件
from typing import Any, Dict

import mlflow
import pandas as pd

from minio import Minio
from minio.error import S3Error
import uuid

import time
import json
import hashlib

logging.basicConfig(level=logging.DEBUG)
DVC_REPO = "https://github.com/chen88088/car-insurance-dataset.git"


SERVER_MANAGER_URL = "http://10.52.52.136:8000"

MACHINE_ID = "machine_server_1"
MACHINE_IP = "10.52.52.136"
MACHINE_PORT = 8085
MACHINE_CAPACITY = 2

class RegisterRequest(BaseModel):
    machine_id: str
    ip: str
    port: int
    capacity: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.debug("Entering lifespan startup")
    register_data = RegisterRequest(
        machine_id=MACHINE_ID,
        ip=MACHINE_IP,
        port=MACHINE_PORT,
        capacity=MACHINE_CAPACITY
    )

    try:
        # #　去跟servermanager註冊自己的ip與資源用量
        # logging.debug("Sending registration request to Server Manager")
        # response = requests.post(f"{SERVER_MANAGER_URL}/register", json=register_data.dict())
        # logging.debug(f"Registration request response status: {response.status_code}")
        # response.raise_for_status()
        logging.info("Registered with Server Manager successfully")
    except requests.RequestException as e:
        logging.error(f"Failed to register with Server Manager: {e}")
        logging.debug(f"Response: {e.response.text if e.response else 'No response'}")

    yield

    logging.debug("Entering lifespan shutdown")
    try:
        # #　去跟servermanager 清除自己的ip與資料
        # response = requests.post(f"{SERVER_MANAGER_URL}/cleanup", json=register_data.dict())
        # response.raise_for_status()
        logging.info("Resources cleaned up successfully")
    except requests.RequestException as e:
        logging.error(f"Failed to clean up resources: {e}")



app = FastAPI(lifespan=lifespan)

# PVC 掛載路徑
STORAGE_PATH = "/mnt/storage"

# # PVC 掛載路徑 (本地測試用)
# STORAGE_PATH = "/mnt/storage/test/test_general"



# MinIO 設定
MINIO_URL = "10.52.52.138:31000"
MINIO_ACCESS_KEY = "testdvctominio"
MINIO_SECRET_KEY = "testdvctominio"
BUCKET_NAME = "mock-dataset"


# 定義 Request Body Schema
class CreateFolderRequest(BaseModel):
    dag_id: str
    execution_id: str

class DagRequest(BaseModel):
    DAG_ID: str
    EXECUTION_ID: str
    TASK_STAGE_TYPE: str
    DATASET_NAME: str
    DATASET_VERSION: str
    DATASET_DVCFILE_REPO: str
    CODE_REPO_URL: Dict[str, str]
    IMAGE_NAME: Dict[str, str]
    EXECUTION_SCRIPTS: Dict[str, List[str]]
    UPLOAD_MLFLOW_SCRIPT: Dict[str, str]
    MODEL_NAME: str
    MODEL_VERSION: str
    DEPLOYER_NAME: str
    DEPLOYER_EMAIL: str
    PIPELINE_CONFIG: Dict[str, Any]

# 檢查 PVC 是否已掛載
def is_pvc_mounted():
    return os.path.exists(STORAGE_PATH) and os.path.ismount(STORAGE_PATH)

# 創建資料夾（如果不存在）
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

# MinIO 客戶端初始化
def init_minio_client():
    return Minio(
        MINIO_URL,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # 如果 MinIO 沒有 SSL，設置為 False
    )

# 解析 DVC 檔案並獲取 outs 路徑
def parse_dvc_file(dvc_file_path):
    with open(dvc_file_path, 'r') as file:
        dvc_data = yaml.safe_load(file)
        outs = dvc_data.get("outs", [])
        dataset_paths = [item['path'] for item in outs]
        return dataset_paths
    
# 初始化 Kubernetes 客戶端
def init_k8s_client():
    try:
        config.load_incluster_config()  # Kubernetes 內部環境
    except config.ConfigException:
        config.load_kube_config()  # 本機開發環境
    return client.BatchV1Api()


# 實例化 LoggerManager
logger_manager = LoggerManager()
# 實例化 DVCManger
dvc_manager = DVCManager(logger_manager)
# 實例化 DagManger
dag_manager = DagManager(logger_manager)

@app.get("/health", status_code=200)
async def health_check():
    return JSONResponse(content={"status": "General_Model_Training_Server is healthy"})

# =========================
# PVC 掛載路徑
# STORAGE_PATH = "/mnt/storage"
CODE_PATH = os.path.join(STORAGE_PATH, "code")
DATASET_PATH = os.path.join(STORAGE_PATH, "dataset")
MODEL_PATH = os.path.join(STORAGE_PATH, "model")
OUTPUT_PATH = os.path.join(STORAGE_PATH, "output")

# =========================
# 型別定義

class DownloadRequest(BaseModel):
    source_url: str
    target_subdir: str  # e.g., "dataset/rice_2023_v1"

class ModifyConfigRequest(BaseModel):
    config_path: str
    updates: dict

class RunJobRequest(BaseModel):
    command: list

class UploadMLflowRequest(BaseModel):
    model_path: str
    metrics_path: str
    run_name: str

class UploadLogRequest(BaseModel):
    log_dir: str
    dag_id: str
    execution_id: str


# =========================

# [Training/RegisterDag]
@app.post("/Training/RegisterDag")
async def register_dag_and_logger_and_dvc_worker(request: DagRequest):
    """
    每隻 DAG 先來註冊，並生成專屬的 logger 與 DVC Worker
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 檢查 PVC 掛載狀態
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")

    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}
    dag_root_folder_path = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")
    git_local_repo_for_dvc = os.path.join(dag_root_folder_path, "GIT_LOCAL_REPO_FOR_DVC")

    # 創建根目錄
    create_folder_if_not_exists(dag_root_folder_path)

    # 確認 DAG 是否已在 dag_manager 中註冊
    if dag_manager.is_registered(dag_id, execution_id):
        return {"status": "success", "message": "DAG is already registered."}

    # 登記 DAG 到 dag_manager
    dag_manager.register_dag(dag_id, execution_id, dag_root_folder_path)

    # 初始化並註冊 logger
    if not logger_manager.logger_exists(dag_id, execution_id):
        logger_manager.init_logger(dag_id, execution_id, dag_root_folder_path)

    # 初始化並註冊 DVCWorker
    create_folder_if_not_exists(git_local_repo_for_dvc)
    dvc_manager.init_worker(dag_id=dag_id, execution_id=execution_id, git_repo_path=git_local_repo_for_dvc)

    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/RegisterDag")

    # **打印請求的 request body**
    request_dict = request.model_dump()
    logger.info(f"Received request body:\n{json.dumps(request_dict, indent=4)}")

    logger.info(f"DAG, Logger, and DVCWorker initialized for DAG_ID: {dag_id}, EXECUTION_ID: {execution_id}")

    return {"status": "success", "message": f"DAG, Logger, and DVCWorker initialized for DAG_ID: {dag_id}, EXECUTION_ID: {execution_id}"}


# [Training/Download_Code_Repo]
@app.post("/Training/Download_Code_Repo")
async def Download_Code_Repo(request: DagRequest):
    """
    為 Training 設定資料夾結構並 clone 相關程式碼
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/SetupFolder")
    
    dvc_worker = dvc_manager.get_worker(dag_id, execution_id)
    
    
    # 檢查 PVC 掛載狀態
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")
    logger.info("PVC IS MOUNTED!!!!")

    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}
    dag_root_folder_path = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}")
    repo_training_path = os.path.join(dag_root_folder_path, "CODE")

    # 確認 DAG 根目錄是否存在
    if not os.path.exists(dag_root_folder_path):
        raise HTTPException(status_code=404, detail="dag_root_folder was not created")


    # 從 CODE_REPO_URL 中獲取對應的 Repo URL
    code_repo_url = request.CODE_REPO_URL.get(task_stage_type)
    if not code_repo_url:
        raise HTTPException(status_code=400, detail="CODE_REPO_URL for the given TASK_STAGE_TYPE not found")

    try:
        logger.info(f"Received request for DAG_ID: {dag_id}, EXECUTION_ID: {execution_id}")

        # 1. 檢查GIT LOCAL REPO FOR DVC
        dvc_worker.ensure_git_repository()
        logger.info("Ensured Git repository for DVC")

        # 2. 在 dag_root_folder 内 git clone
        if not os.path.exists(repo_training_path):
            # 取得環境變數中的 GITHUB_TOKEN
            github_token = os.getenv("GITHUB_TOKEN") 
            if not github_token:
                raise Exception("GITHUB_TOKEN not found in environment variables.")

            # 使用 subprocess.run() 執行 git clone 指令
            # **修正：直接內嵌 GITHUB_TOKEN 到 URL 中**
            # repo_url = f"https://{github_token}:x-oauth-basic@github.com/chen88088/NCU-RSS-1.5.git"
            repo_url = code_repo_url.replace("https://", f"https://{github_token}:x-oauth-basic@")
            clone_command = ["git", "clone", repo_url, repo_training_path]
            result = subprocess.run(clone_command, capture_output=True, text=True)
            
            # 紀錄輸出與錯誤訊息
            logger.info(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")

            # 檢查返回碼
            if result.returncode != 0:
                raise Exception("git clone failed")
            
            # 驗證是否成功 clone
            assert os.path.exists(repo_training_path), "Repo Training Code clone failed"
            logger.info(f"Cloned NCURSS-Training repo to {repo_training_path}")
        else:
            logger.info(f"NCURSS-Training repo already exists at {repo_training_path}")

        return {"status": "success",  "message": f"Cloned NCURSS-Training repo to {repo_training_path}"}

    except HTTPException as e:
        logger.error(f"HTTPException occurred: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API: 下載資料集
# [Training/DownloadDataset]
@app.post("/Training/DownloadDataset")
async def download_dataset(request: DagRequest):
    """
    從 MinIO 下載資料集至 PVC 中
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    dataset_dvcfile_repo = request.DATASET_DVCFILE_REPO

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/DownloadDataset")

    # 檢查 PVC 掛載狀態
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")

    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}/data
    target_folder = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}", "DATA")
    create_folder_if_not_exists(target_folder)

    try:

        # 取得環境變數中的 GITHUB_TOKEN
        github_token = os.getenv("GITHUB_TOKEN") 
        if not github_token:
            raise Exception("GITHUB_TOKEN not found in environment variables.")

        # Step 1: clone DVC repo 到 /mnt/storage/{dag_id}_{execution_id}/DATA/
        # 使用 subprocess.run() 執行 git clone 指令
        # **修正：直接內嵌 GITHUB_TOKEN 到 URL 中**
        # repo_url = f"https://{github_token}:x-oauth-basic@github.com/chen88088/NCU-RSS-1.5.git"
        dataset_dvcfile_repo_url = dataset_dvcfile_repo.replace("https://", f"https://{github_token}:x-oauth-basic@")
        clone_command = ["git", "clone", dataset_dvcfile_repo_url, target_folder]
        subprocess.run(clone_command, capture_output=True, text=True)

        # Step 2: 在該資料夾下執行 dvc pull
        subprocess.run(["dvc", "pull"], cwd=target_folder, check=True)

        # Step 3: 搬檔案出來
        def flatten_nested_data_folder(target_folder: str):
            nested_data_path = os.path.join(target_folder, "DATA")

            if os.path.isdir(nested_data_path):
                temp_path = os.path.join(os.path.dirname(target_folder), "__temp_data")
                os.rename(nested_data_path, temp_path)
                shutil.rmtree(target_folder)  # 此時 target_folder 沒有檔案了
                os.rename(temp_path, target_folder)
        
        flatten_nested_data_folder(target_folder)

        logger.info(f"DVC dataset pulled into: {target_folder}")
        return {
            "status": "success",
            "message": f"Dataset downloaded to {target_folder}"
        }
        

        # # 初始化 MinIO 客戶端
        # minio_client = dvc_worker.init_minio_client()
        # logger.info("MinIO Client Initialization Succesffully")

        # # 1. 下載 dvc_file 中的 result.dvc
        # dvc_worker.download_dvc_file(minio_client, target_folder)
        # logger.info("Dataset .dvc File download successfully")

        # # 2. 使用 DVC Pull 下載 dataset
        # dvc_worker.download_dataset_with_dvc(target_folder)
        # logger.info("Download Dataset with .dvc file successfully")
        # # 3. 下載 excel_file 資料夾
        # dvc_worker.download_excel_files(minio_client, target_folder)
        # logger.info("Download Excel Files successfully")
        # # **4. 重組資料夾結構**
        # dvc_worker.reorganize_data_folder(target_folder)
        # logger.info("Download Excel Files successfully")

        # logger.info(f"Dataset and Excel files downloaded to {target_folder}")

        return {"status": "success", "message": f"Dataset and Excel files downloaded to {target_folder}"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.post("/Training/AddConfig")
async def add_config(request: DagRequest):
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    pipeline_config = request.PIPELINE_CONFIG

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    if not pipeline_config or not isinstance(pipeline_config, dict):
        raise HTTPException(status_code=400, detail="PIPELINE_CONFIG must be provided as a dict.")

    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/AddConfig")
    
    if not is_pvc_mounted():
        raise HTTPException(status_code=500, detail="PVC is not mounted.")

    config_dir = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}", "CONFIG")
    create_folder_if_not_exists(config_dir)

    logger.info(f"CONFIG folder created!!: {config_dir}")

    config_path = os.path.join(config_dir, "config.yaml")

    try:
        with open(config_path, "w") as f:
            yaml.dump(pipeline_config, f, default_flow_style=False)

        logger.info(f"Config file saved to: {config_path}")
        return {
            "status": "success",
            "message": f"Config created at {config_path}"
        }

    except Exception as e:
        logger.error(f"Failed to write config: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# [Training/ExecuteTrainingScripts]
@app.post("/Training/ExecuteTrainingScripts")
async def execute_training_scripts(request: DagRequest):
    """
    根據 TASK_STAGE_TYPE 起 Pod 並執行對應腳本
    """
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    # 從 IMAGE_NAME 中抓取對應的 Image
    image_name = request.IMAGE_NAME.get(task_stage_type)
    if not image_name:
        raise HTTPException(status_code=400, detail=f"No image found for TASK_STAGE_TYPE: {task_stage_type}")
    
    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/ExecuteTrainingScripts")

    output_dir = os.path.join(STORAGE_PATH, f"{dag_id}_{execution_id}", "OUTPUT")
    create_folder_if_not_exists(output_dir)
    

    # v1 = init_k8s_client()
    batch_v1 = init_k8s_client() # 初始化 Batch API 用於創建 Job
    
    # # Pod 名稱
    # pod_name = f"{dag_id}-{execution_id}-{task_stage_type}-task-pod-{uuid.uuid4().hex[:6]}"

    # Job 名稱
    raw_job_name = f"task-job-{dag_id}-{execution_id}-{task_stage_type}-{uuid.uuid4().hex[:6]}"
    # 將 _ 換成 -，轉小寫，並移除不合法字元
    job_name = re.sub(r'[^a-z0-9\-]+', '', raw_job_name.lower().replace('_', '-'))

    # 長度限制處理（K8s label/name 最多 63 字元）
    if len(job_name) > 63:
        hash_suffix = hashlib.sha1(job_name.encode()).hexdigest()[:6]
        job_name = f"{job_name[:56]}-{hash_suffix}"

    
    # 確保 PVC_NAME 環境變量已設定
    pvc_name = os.getenv("PVC_NAME")
    if not pvc_name:
        raise HTTPException(status_code=500, detail="PVC_NAME not set in environment variables")
    
    # 組合工作目錄
    working_dir = f"/mnt/storage/{dag_id}_{execution_id}/CODE"
    
    # 定義要執行的腳本
    # 從 SCRIPTS 中抓取該階段要執行的 script list
    script_filenames = request.EXECUTION_SCRIPTS.get(task_stage_type)
    if not script_filenames:
        raise HTTPException(status_code=400, detail=f"No scripts found for TASK_STAGE_TYPE: {task_stage_type}")

    # 自動加上 python3 執行前綴
    scripts_to_run = [f"python3 {script}" for script in script_filenames]


    # 組合成 Command
    command = " && ".join([f"cd {working_dir}"] + scripts_to_run )
    
    # **正確的 Job Manifest**
    job_manifest = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "ml-serving",
            "labels": {
                "app": "task-job",
                "type": "gpu"
            }
        },
        "spec": {
            "backoffLimit": 3,
            # **TTLAfterFinished: 完成後 60 秒自動刪除**
            "ttlSecondsAfterFinished": 60,
            "template": {
                "metadata": {
                    "name": job_name
                },
                "spec": {
                    "nodeSelector": {
                        "gpu-node": "true"
                    },
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "task-container",
                            "image": f"harbor.pdc.tw/{image_name}",
                            "command": ["/bin/bash", "-c", command],
                            "volumeMounts": [
                                {
                                    "name": "shared-storage",
                                    "mountPath": "/mnt/storage"
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "nvidia.com/gpu": "1"
                                }
                            }
                        }
                    ],
                    "volumes": [
                        {
                            "name": "shared-storage",
                            "persistentVolumeClaim": {
                                "claimName": pvc_name
                            }
                        }
                    ]
                }
            }
        }
    }
    
    # 建立 Job
    try:
        logger.info("Job creating...........")
        batch_v1.create_namespaced_job(namespace="ml-serving", body=job_manifest)

        logger.info("Job execute start...........")
        # **等待 Job 完成**
        wait_for_job_completion(batch_v1, job_name, "ml-serving",logger)

        logger.info("Job finish!!!!!")
        
        # output_dir = f"{working_dir}/data/train_test/For_training_testing/320x320/train_test"
        # # record_mlflow(request, output_dir)
        # # logger.info("Record to MLflow Successfully!!!!")
        logger.info(f"Task Job created and finished . job_name {job_name}")
        return {"status": "success", "message": "Task Job created and finished", "job_name": job_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create Task Job: {str(e)}")

def wait_for_job_completion(batch_v1, job_name, namespace, logger, timeout=3600):
    """
    監聽 K8s Job 狀態，等待 Job 執行完成
    """
    start_time = time.time()
    v1 = client.CoreV1Api()  # 初始化 K8s API
    pod_name = None  # 儲存 task pod 名稱
    while time.time() - start_time < timeout:
        job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
        if job_status.status.succeeded == 1:
            logger.info(f"Job {job_name} completed successfully.")
            
            # **讀取 Pod 名稱**
            pod_list = v1.list_namespaced_pod(namespace, label_selector=f"job-name={job_name}").items
            
            if pod_list:
                pod_name = pod_list[0].metadata.name
                logger.info(f"Job pod name: {pod_name}")

                # **獲取並記錄 Pod 的 logs**
                try:
                    logs = v1.read_namespaced_pod_log(name=pod_name, namespace=namespace)
                    logger.info(f"Task Pod Logs for {job_name}:\n{logs}")
                except Exception as e:
                    logger.error(f"Failed to get logs for {pod_name}: {str(e)}")

            return
        
        elif job_status.status.failed is not None and job_status.status.failed > 0:
            raise Exception(f"Job {job_name} failed.")
        time.sleep(10)  # 每 10 秒檢查一次 Job 狀態


@app.post("/Training/upload_mlflow")
def upload_mlflow(request: DagRequest):
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/upload_mlflow")

    upload_mlflow_script_filename = request.UPLOAD_MLFLOW_SCRIPT.get(task_stage_type)
    if not upload_mlflow_script_filename:
        raise HTTPException(status_code=400, detail=f"No script filename for TASK_STAGE_TYPE: {task_stage_type}")

    working_dir = f"/mnt/storage/{dag_id}_{execution_id}"

    # 找到 code repo 中的 upload_mlflow script
    script_path = os.path.join(working_dir, "CODE" , "upload_mlflow", upload_mlflow_script_filename)
    if not os.path.exists(script_path):
        raise HTTPException(status_code=404, detail=f"Script not found at path: {script_path}")
    
    logger.info(f"upload mlflow script found: {script_path}")

    experiment_name = dag_id
    experiment_run_name = f'{dag_id}_{execution_id}'

    # 把關鍵資訊傳進腳本，剩下都由腳本內部自行處理
    command = [
        "python3", script_path,
        experiment_name,
        experiment_run_name
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Script failed: {result.stderr}")
    
    logger.info(f"upload mlflow execution successfully !!!")
    
    return {
        "status": "success",
        "stdout": result.stdout
    }

# [Training/UploadLogToS3]
@app.post("/Training/UploadLogToS3")
async def upload_log_to_s3(request: DagRequest):
    
    dag_id = request.DAG_ID
    execution_id = request.EXECUTION_ID
    task_stage_type = request.TASK_STAGE_TYPE

    if not dag_id or not execution_id:
        raise HTTPException(status_code=400, detail="DAG_ID and EXECUTION_ID are required.")
    
    # 組合路徑：/mnt/storage/{dag_id}_{execution_id}/LOGS
    storage_path = STORAGE_PATH
    logs_folder_path = Path(storage_path) / f"{dag_id}_{execution_id}" / "LOGS"

    # 獲取 Logger 和 DVCWorker
    logger = logger_manager.get_logger(dag_id, execution_id)
    if logger:
        logger_manager.log_section_header(logger, "Training/UploadLogToS3")
    dvc_worker = dvc_manager.get_worker(dag_id, execution_id)

    try:
        
        # Log 檔案路徑
        log_file_path = logs_folder_path / f"{dag_id}_{execution_id}.txt"

        if not log_file_path.exists():
            raise HTTPException(status_code=404, detail=f"Log file {log_file_path} not found.")

        # 重新命名檔案
        renamed_log_filename = f"{dag_id}_{execution_id}_{task_stage_type}.txt"

        # 上傳到 MinIO
        target_path = f"{dag_id}_{execution_id}/logs/{renamed_log_filename}"

        logger.info(f"Uploading log file: {log_file_path} to MinIO at {target_path}")

        # 使用 DVCWorker 的 S3 客戶端上傳
        dvc_worker.s3_client.upload_file(
            Filename=str(log_file_path),
            Bucket=dvc_worker.minio_bucket,
            Key=target_path
        )

        logger.info("Log file uploaded successfully.")

        return {
            "status": "success",
            "message": f"Log file uploaded to MinIO at {target_path}"
        }

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}") 


# #############################################################
# @app.post("/download_dataset")
# def download_dataset(req: DownloadRequest):
#     target_dir = os.path.join(STORAGE_PATH, req.target_subdir)
#     os.makedirs(target_dir, exist_ok=True)
#     try:
#         subprocess.run(["wget", "-q", req.source_url, "-P", target_dir], check=True)
#     except subprocess.CalledProcessError:
#         raise HTTPException(status_code=500, detail="Failed to download dataset.")
#     return {"message": "Dataset downloaded to " + target_dir}

# @app.post("/download_code_repo")
# def download_code_repo(req: DownloadRequest):
#     target_dir = os.path.join(STORAGE_PATH, req.target_subdir)
#     if os.path.exists(target_dir):
#         shutil.rmtree(target_dir)
#     try:
#         subprocess.run(["git", "clone", req.source_url, target_dir], check=True)
#     except subprocess.CalledProcessError:
#         raise HTTPException(status_code=500, detail="Failed to clone code repo.")
#     return {"message": "Code repo cloned to " + target_dir}

# @app.post("/modify_config")
# def modify_config(req: ModifyConfigRequest):
#     try:
#         with open(req.config_path, "r") as f:
#             cfg = yaml.safe_load(f)
#         cfg.update(req.updates)
#         with open(req.config_path, "w") as f:
#             yaml.dump(cfg, f)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to modify config: {e}")
#     return {"message": "Config updated."}

# @app.post("/run_job")
# def run_job(req: RunJobRequest):
#     try:
#         result = subprocess.run(req.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         return {
#             "returncode": result.returncode,
#             "stdout": result.stdout,
#             "stderr": result.stderr
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Job failed to run: {e}")

# @app.post("/upload_mlflow")
# def upload_mlflow(req: UploadMLflowRequest):
#     try:
#         result = subprocess.run([
#             "python", os.path.join(CODE_PATH, "upload", "upload_to_mlflow.py"),
#             "--model", req.model_path,
#             "--metrics", req.metrics_path,
#             "--run_name", req.run_name
#         ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#         return {
#             "returncode": result.returncode,
#             "stdout": result.stdout,
#             "stderr": result.stderr
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to upload to MLflow: {e}")

# @app.post("/upload_log")
# def upload_log(req: UploadLogRequest):
#     try:
#         zip_filename = f"logs_{req.dag_id}_{req.execution_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
#         zip_path = os.path.join(OUTPUT_PATH, zip_filename)
#         shutil.make_archive(zip_path.replace(".zip", ""), 'zip', req.log_dir)
#         return {"message": f"Log archived to {zip_path}"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to archive log: {e}")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("MachineServer:app", host=MACHINE_IP, port=MACHINE_PORT, reload=True)



