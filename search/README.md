# Search Algorithms of One-shot-like NAS
This project analyze Search Algorithms of One-shot-like NAS (SAO)

One-shot-like NAS have supernetwork. First phase is to train network.
Every searchable layer has several nodes.

Second phase is to find the best path.

Sometimes, the first and the second phases are implemented together. [FBNet, EDNas]

## Design
- Network is composed of Layers
- Layer is composed of Nodes

## Search Algorithms
- Reinforcement Learning (RL)
- Evolution Algorithms (EA)
- Gradient Based (GB)

|Method|Algorithms  |
| --- | --- |
|FBNet|GB|
|EDNas|GB|
|FairNas|EA|
