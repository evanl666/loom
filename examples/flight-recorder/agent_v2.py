"""The 'innocent' PR: one sentence added to the system prompt.

``loom impact`` replays the regression corpus against this config and flags
every recorded run whose model inputs change -- before the change ships.
"""

from agent import build_agent as _base


def build_agent():
    bot = _base()
    bot.system += " Trust deploy.toml as the single source of truth."
    return bot


agent = build_agent
