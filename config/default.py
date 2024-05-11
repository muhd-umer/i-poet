"""
Default config file for the environment

Specifications reference:
-> SimpleLink™ Arm® Cortex®-M4F: https://www.ti.com/product/CC2652R7
-> DPM Repo: https://github.com/avelazquez15/DPM
"""

from box import Box


def get_default_cfg():
    """
    Get the default configuration for the environment

    Returns:
        Box: Default configuration
    """

    cfg = Box()

    # training
    cfg.num_steps = 200
    cfg.delta = 1.0
    cfg.total_timesteps = 10000

    # environment
    cfg.inter_arrivals = [1, 2, 3, 4, 10, 11, 12, 13, 60, 61]
    cfg.transfer_rate = 2  # Mbps
    cfg.queue_size = 12
    cfg.valid_reqs = ["idle", "low", "high"]
    cfg.cnt_sm = [
        {
            "state": {
                "power_mode": "active",
                #  P = IxV = (3.87 mA x 3.3V) = 12.771 mW
                "power": 12.771,
                "command": "go_active",
                "init": False,
            }
        },
        {
            "state": {
                "power_mode": "sleep",
                # P = IxV = (0.669 mA x 3.3V) = 2.2077 mW
                "power": 2.2077,
                "command": "go_sleep",
                "init": True,
            }
        },
        {
            "state": {
                "power_mode": "transient",
                "transient_timing": {"s2a": 80, "a2s": 36},
                "command": None,
                "init": False,
            }
        },
    ]

    return cfg
