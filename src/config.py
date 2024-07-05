from dataclasses import dataclass


@dataclass
class LyngorBuildFlags:
    '''
        SeamlessStreamingS2STJointVADAgent(
                SileroVADAgent[speech -> speech]
                OnlineFeatureExtractorAgent[speech -> speech]
                OfflineWav2VecBertEncoderAgent[speech -> speech]
                UnitYMMATextDecoderAgent[speech -> text]
                UnitYDetokenizerAgent[text -> text]
                NARUnitYUnitDecoderAgent[text -> text]
                VocoderAgent[text -> speech]
        )
    '''
    build_offlineWav2VecBertEncoderAgent: str = ["False", "True"][0]
    build_vocoderAgent: str = ["False", "True"][0]

    def init_os_env(self):
        import os
        env_mapping = {
            "BUILD_OFFLINEWAV2VECBERTENCODERAGENT": self.build_offlineWav2VecBertEncoderAgent,
            "BUILD_VOCODERAGENT": self.build_vocoderAgent,
        }
        for k, v in env_mapping.items():
            os.environ[k] = v
            print(f'{k}:', os.environ.get(k))


@dataclass
class ModelSaveWeightFlags:
    '''
        SeamlessStreamingS2STJointVADAgent(
                SileroVADAgent[speech -> speech]
                OnlineFeatureExtractorAgent[speech -> speech]
                OfflineWav2VecBertEncoderAgent[speech -> speech]
                UnitYMMATextDecoderAgent[speech -> text]
                UnitYDetokenizerAgent[text -> text]
                NARUnitYUnitDecoderAgent[text -> text]
                VocoderAgent[text -> speech]
        )
    '''
    save_offlineWav2VecBertEncoderAgent: str = ["False", "True"][0]
    save_vocoderAgent: str = ["False", "True"][0]

    def init_os_env(self):
        import os
        env_mapping = {
            "SAVE_OFFLINEWAV2VECBERTENCODERAGENT": self.save_offlineWav2VecBertEncoderAgent,
            "SAVE_VOCODERAGENT": self.save_vocoderAgent,
        }
        for k, v in env_mapping.items():
            os.environ[k] = v
            print(f'{k}:', os.environ.get(k))
