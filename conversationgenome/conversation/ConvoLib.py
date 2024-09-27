import requests

from conversationgenome.utils.Utils import Utils
from conversationgenome.ConfigLib import c

from conversationgenome.api.ApiLib import ApiLib


class ConvoLib:
    async def get_conversation(self, hotkey, api_key=None):
        api = ApiLib()
        convo = await api.reserveConversation(hotkey, api_key=api_key)
        return convo

    async def put_conversation(self, hotkey, c_guid, data, type="validator", batch_num=None, window=None):
        llm_type = "openai"
        model = "gpt-4o"
        llm_type_override = c.get("env", "LLM_TYPE_OVERRIDE")
        if llm_type_override:
            llm_type = llm_type_override
            model = c.get("env", "OPENAI_MODEL")
        llm_model = c.get('env', llm_type.upper() + "_MODEL")
        output = {
            "type": type,
            "mode": c.get('env', 'SYSTEM_MODE'),
            "model": llm_model,
            "marker_id": c.get('env', 'MARKER_ID'),
            "convo_window_index": window,
            "hotkey": hotkey,
            "llm_type" : c.get('env', 'LLM_TYPE'),
            "scoring_version" : c.get('system', 'scoring_version'),
            "batch_num" : batch_num,
            "cgp_version": "0.1.0",
            "netuid": c.get("system", "netuid"),
            "data": data,
        }
        api = ApiLib()
        result = await api.put_conversation_data(c_guid, output)
        return result
