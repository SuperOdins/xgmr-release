"""
Ablation study dispatcher.

한국어: 데코레이터를 이용해 원본 로직과 어블레이션 로직을 분기한다.
English: Dispatches between original logic and ablation variants using decorators.
"""

import sys
import dataclasses
from typing import Any, Callable

_GLOBAL_CONFIG = None


def set_ablation_config(cfg: Any) -> None:
    """
    글로벌 어블레이션 설정을 주입한다.
    Korean: 데코레이터가 실시간으로 설정을 참조할 수 있도록 전역 변수에 저장한다.
    """
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = cfg
    if cfg is not None:
        print(f"[Ablation] Global config registered. (Mode: {get_ablation_slug(cfg.ablation)})")

def get_ablation_slug(config: Any) -> str:
    """
    AblationConfig 객체로부터 실험 이름 문자열(Slug)을 생성합니다.
    - 활성화된 플래그가 없으면 "Full" 반환.
    - 모든 플래그가 활성화되어 있으면 "Vanilla" 반환.
    - 그 외에는 활성화된 플래그들을 조합하여 "w_o_eba_lgeo" 형태로 반환.
    """
    if config is None:
        return "Unknown"
        
    active_tags = []
    all_ab_fields = []
    
    # dataclass 필드를 순회하며 활성화된 no_* 플래그 확인
    for field in dataclasses.fields(config):
        if field.name.startswith("no_"):
            all_ab_fields.append(field.name)
            if getattr(config, field.name):
                tag = field.name.replace("no_", "")
                active_tags.append(tag)
    
    if not active_tags:
        return "Full"
    
    if len(active_tags) == len(all_ab_fields) and len(all_ab_fields) > 0:
        return "Vanilla"
        
    return "w_o_" + "_".join(active_tags)


def ablation_entry(option: str) -> Callable:
    """
    어블레이션 진입점 데코레이터.
    Korean: 지정된 옵션(예: 'no_eba')이 True이면 {함수명}_ablation을 호출하고, False이면 원본을 호출한다.
    English: If the specified option is True, calls {func_name}_ablation; otherwise calls the original function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 0. 재귀 호출 방지 (Ablation 함수가 원본 함수를 다시 호출하는 경우)
            if getattr(wrapper, "_in_ablation", False):
                return func(*args, **kwargs)

            # 설정이 존재하고 해당 옵션이 True인 경우 어블레이션 함수 탐색
            if _GLOBAL_CONFIG is not None and getattr(_GLOBAL_CONFIG.ablation, option, False):
                ablation_func_name = f"{func.__name__}_ablation"
                
                # 1. 클래스 메서드인 경우 self(첫 번째 인자)를 통해 탐색 시도
                if args and hasattr(args[0], ablation_func_name):
                    ablation_func = getattr(args[0], ablation_func_name)
                    wrapper._in_ablation = True
                    try:
                        return ablation_func(*args[1:], **kwargs)
                    finally:
                        wrapper._in_ablation = False

                # 2. 일반 함수인 경우 모듈 레벨에서 탐색 시도
                module = sys.modules[func.__module__]
                if hasattr(module, ablation_func_name):
                    ablation_func = getattr(module, ablation_func_name)
                    wrapper._in_ablation = True
                    try:
                        return ablation_func(*args, **kwargs)
                    finally:
                        wrapper._in_ablation = False

            return func(*args, **kwargs)

        wrapper._in_ablation = False
        return wrapper

    return decorator
