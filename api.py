from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
import librosa
import numpy as np
import soundfile as sf
import os
import uuid
from lib import spec_utils, nets
import shutil
from typing import Optional

app = FastAPI(
    title="음성 분리 API",
    description="""
    음악 파일에서 보컬과 반주를 분리하는 API입니다.
    
    ## 주요 기능
    * 음악 파일 업로드 및 분리
    * 보컬/반주 개별 다운로드
    * GPU 가속 지원
    * TTA(Test Time Augmentation) 지원
    
    ## 사용 방법
    1. `/separate/` 엔드포인트에 음악 파일을 업로드
    2. 반환된 URL을 통해 분리된 파일 다운로드
    """,
    version="1.0.0"
)

# 전역 변수로 모델 초기화
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'baseline.pth')

# 임시 파일 저장을 위한 디렉토리
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# 모델 초기화 함수
def init_model(gpu=-1):
    device = torch.device('cpu')
    if gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu}')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    
    model = nets.CascadedNet(2048, 1024, 32, 128)
    model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location='cpu'))
    model.to(device)
    return model, device

# 전역 변수로 모델 로드
model, device = init_model()

@app.post("/separate/", 
    summary="음악 파일 분리",
    description="""
    업로드된 음악 파일을 보컬과 반주로 분리합니다.
    
    - 지원 파일 형식: WAV
    - 권장 샘플링 레이트: 44.1kHz
    - 스테레오/모노 모두 지원
    """,
    response_description="분리된 파일의 다운로드 URL"
)
async def separate_audio(
    file: UploadFile = File(..., description="분리할 음악 파일 (WAV 형식)"),
    gpu: int = -1,
    tta: bool = False,
    post_process: bool = False
):
    """
    Parameters:
    - file: 분리할 음악 파일
    - gpu: 사용할 GPU 번호 (-1: CPU 사용)
    - tta: Test Time Augmentation 사용 여부
    - post_process: 후처리 적용 여부
    """
    try:
        # 임시 파일 경로 생성
        temp_id = str(uuid.uuid4())
        input_path = os.path.join(TEMP_DIR, f"input_{temp_id}.wav")
        output_inst_path = os.path.join(TEMP_DIR, f"output_inst_{temp_id}.wav")
        output_vocal_path = os.path.join(TEMP_DIR, f"output_vocal_{temp_id}.wav")

        # 업로드된 파일 저장
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 오디오 로드
        X, sr = librosa.load(
            input_path, sr=44100, mono=False, dtype=np.float32, res_type='kaiser_fast'
        )

        if X.ndim == 1:
            X = np.asarray([X, X])

        # 스펙트로그램 변환
        X_spec = spec_utils.wave_to_spectrogram(X, 1024, 2048)

        # Separator 설정
        from inference import Separator
        sp = Separator(
            model=model,
            device=device,
            batchsize=4,
            cropsize=256,
            postprocess=post_process
        )

        # 분리 실행
        if tta:
            y_spec, v_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec = sp.separate(X_spec)

        # 결과 저장
        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=1024)
        sf.write(output_inst_path, wave.T, sr)

        wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=1024)
        sf.write(output_vocal_path, wave.T, sr)

        # 결과 반환
        return {
            "instrumental": f"/download/instrumental/{temp_id}",
            "vocal": f"/download/vocal/{temp_id}"
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/download/instrumental/{temp_id}",
    summary="반주 파일 다운로드",
    description="분리된 반주 파일을 다운로드합니다.",
    response_description="WAV 형식의 반주 파일"
)
async def download_instrumental(temp_id: str):
    """
    Parameters:
    - temp_id: separate 엔드포인트에서 반환된 임시 ID
    """
    file_path = os.path.join(TEMP_DIR, f"output_inst_{temp_id}.wav")
    return FileResponse(file_path, filename="instrumental.wav", media_type="audio/wav")

@app.get("/download/vocal/{temp_id}",
    summary="보컬 파일 다운로드",
    description="분리된 보컬 파일을 다운로드합니다.",
    response_description="WAV 형식의 보컬 파일"
)
async def download_vocal(temp_id: str):
    """
    Parameters:
    - temp_id: separate 엔드포인트에서 반환된 임시 ID
    """
    file_path = os.path.join(TEMP_DIR, f"output_vocal_{temp_id}.wav")
    return FileResponse(file_path, filename="vocal.wav", media_type="audio/wav")

# 서버 종료 시 임시 파일 정리를 위한 이벤트 핸들러
@app.on_event("shutdown")
def cleanup():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 