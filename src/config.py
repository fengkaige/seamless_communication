from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OSFlags:
    env_mapping: Dict = field(default_factory=dict)
    meaning_mapping: Dict = field(default_factory=dict)

    def init_os_env(self):
        """初始化系统环境变量

        Returns:
            None

        Raises:
            ValueError: env_mapping为空
        """
        if not self.env_mapping:
            raise ValueError(
                "env_mapping is empty, please call add_os_env() to add os env mapping."
            )
        print("-" * 6, __class__.__name__, "init_os_env", "-" * 6)
        import os

        for k, v in self.env_mapping.items():
            os.environ[k] = v
            print(f"{k}:", os.environ.get(k))

    def set_os_env_mapping(
        self,
        os_env_name: str,
        attribute_name: str,
        default_value: str,
        meaning: str = "",
    ):
        """添加系统环境变量的映射
        映射规则:
            - 系统环境变量名: os_env_name
            - 环境变量值: self.__getattribute__(attribute_name)

        Args:
            os_env_name (str): 系统环境变量名
            attribute_name (str): 环境变量值对应的属性名
            default_value (str): 环境变量值的默认值
            meaning (str, optional): 环境变量值的含义, 默认为"".
        """
        import os

        # init attr
        self.__setattr__(attribute_name, default_value)
        # set mapping
        self.env_mapping.update({os_env_name: self.__getattribute__(attribute_name)})
        self.meaning_mapping.update({os_env_name: meaning})
        # set os env
        os.environ[os_env_name] = self.__getattribute__(attribute_name)

    def set_os_env(
        self,
        os_env_name: str,
        value: str,
        meaning: str = "",
    ):
        """添加系统环境变量的映射
        映射规则:
            - 系统环境变量名: os_env_name
            - 环境变量值: value

        Args:
            os_env_name (str): 系统环境变量名
            default_value (str): 环境变量值的值
            meaning (str, optional): 环境变量值的含义, 默认为"".
        """
        self.set_os_env_mapping(
            os_env_name=os_env_name,
            attribute_name=os_env_name.lower(),
            default_value=value,
            meaning=meaning,
        )

    def __str__(self) -> str:
        _str = f"{self.__class__.__name__}:\n"
        _str += f"\tenv_mapping:\n"
        for k, v in self.env_mapping.items():
            _str += f"\t\t{k}: {v}\n"
        _str += f"\tmeaning_mapping:\n"
        for k, v in self.meaning_mapping.items():
            _str += f"\t\t{k}: {v}\n"
        return _str


@dataclass
class LyngorBuildFlags(OSFlags):
    """
    SeamlessStreamingS2STJointVADAgent(
            SileroVADAgent[speech -> speech]
            OnlineFeatureExtractorAgent[speech -> speech]
            OfflineWav2VecBertEncoderAgent[speech -> speech]
            UnitYMMATextDecoderAgent[speech -> text]
            UnitYDetokenizerAgent[text -> text]
            NARUnitYUnitDecoderAgent[text -> text]
            VocoderAgent[text -> speech]
    )
    """

    ...


@dataclass
class ModelSaveWeightFlags(OSFlags):
    """
    SeamlessStreamingS2STJointVADAgent(
            SileroVADAgent[speech -> speech]
            OnlineFeatureExtractorAgent[speech -> speech]
            OfflineWav2VecBertEncoderAgent[speech -> speech]
            UnitYMMATextDecoderAgent[speech -> text]
            UnitYDetokenizerAgent[text -> text]
            NARUnitYUnitDecoderAgent[text -> text]
            VocoderAgent[text -> speech]
    )
    """

    ...
