# Server

Mert 和 MEL 的后端。请运行 `mert_server.py`。如果前者不能使用 ipv6 访问 Wikidata，则需要在支持 ipv6 的服务器上运行  `wiki_server.py`。

分为控制层和服务层。

## `mert_server`, `mert_service`

用于和模型交互。

服务端参数：

- `-a`, `--address` 设置地址。默认为 `0.0.0.0`。
- `-p`, `--port` 设置端口。默认为 `5001`。已知端口为 3001 会出现加载模型后无法访问 api 的问题。
- `-w`, `-wiki` 在本服务器访问 wiki。

## `wiki_server`, `wiki_service`

用于从 Wikidata 获取 entity 的数据。

服务端参数：

- `-a`, `--address` 设置地址。默认为 `0.0.0.0`。
- `-p`, `--port` 设置端口。默认为 `3002`。