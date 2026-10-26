// Eva's World: Mindcraft settings. apply_patches.sh copies this over
// mindcraft/settings.js before every world start, so edit THIS file
// (interface/externals/ai_minecraft/nova/config/) and redeploy.
//
// Differences from Mindcraft's stock settings, and why:
//   host/port        the local Paper server (offline mode, Tailscale address)
//   profiles         one bot, eva_body.json, driven by Gemma on nova
//   load_memory      true: the body remembers across sessions, like Eva does
//   auto_open_ui     false: nova is headless; the hub's monitor is the UI
//   render_bot_view  true: the 3D viewer, proxied by the hub at /viewer/
//   blocked_actions  + !searchWiki (fetches minecraft.wiki: no outbound
//                    requests from the world) + !newAction (code execution)
//   spawn_timeout    60: the first spawn on a fresh world generates chunks
const settings = {
    "minecraft_version": "1.21.6",
    "host": "@BIND_IP@",
    "port": 25565,
    "auth": "offline",

    "mindserver_port": 8080,
    "auto_open_ui": false,

    "base_profile": "survival",
    "profiles": [
        "./eva_body.json",
    ],

    "load_memory": true,
    "init_message": "",
    "only_chat_with": [],

    "speak": false,
    "chat_ingame": true,
    "language": "en",
    "render_bot_view": true,

    "allow_insecure_coding": false,
    "allow_vision": false,
    "blocked_actions" : ["!checkBlueprint", "!checkBlueprintLevel", "!getBlueprint", "!getBlueprintLevel", "!searchWiki", "!newAction"],
    "code_timeout_mins": -1,
    "relevant_docs_count": 5,

    "max_messages": 15,
    "num_examples": 2,
    "max_commands": -1,
    "show_command_syntax": "full",
    "narrate_behavior": true,
    "chat_bot_messages": true,

    "spawn_timeout": 60,
    "block_place_delay": 0,

    "log_all_prompts": false,
};

export default settings;
