import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken='')
dl_list_file_entities = syn.get_download_list('/path/to/kcd_data/Downloads/btcv')