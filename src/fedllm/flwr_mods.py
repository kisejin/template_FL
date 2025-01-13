from flwr.common import Context, Message, MessageType, ConfigsRecord
from flwr.client.typing import ClientAppCallable
from typing import Callable
import wandb
import time
from .myfedavg import client_id_idx


# Define type alias for Mod
Mod = Callable[[Message, Context, ClientAppCallable], Message]

def get_wandb_mod(name: str) -> Mod:
    # Keep track of active runs
    active_run: Optional[wandb.Run] = None
    
    def wandb_mod(msg: Message, context: Context, app: ClientAppCallable) -> Message:
        nonlocal active_run
        server_round = int(msg.metadata.group_id)
        run_id = msg.metadata.run_id
        group_name = f"Run ID: {run_id}"
        node_id = str(msg.metadata.dst_node_id)
        run_name = f"Client ID: {client_id_idx[node_id]}"
        
        wandb.init(
                project=name,
                group=group_name,
                name=run_name,
                id=f"{run_id}_{client_id_idx[node_id]}",
                resume="allow",
                reinit=True,
                # settings=wandb.Settings(start_method="thread")
        )

        start_time = time.time()
        reply = app(msg, context)

        if reply.metadata.message_type == MessageType.TRAIN and reply.has_content():

            time_diff = time.time() - start_time
            metrics = reply.content.configs_records
            results_to_log = dict(metrics.get("fitres.metrics", ConfigsRecord()))
            results_to_log["fit_time"] = time_diff
            
            wandb.log(results_to_log, step=int(server_round), commit=True)

        return reply

    return wandb_mod

