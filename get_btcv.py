import synapseclient
syn = synapseclient.Synapse()
syn.login(authToken='')
dl_list_file_entities = syn.get_download_list('/Users/mcgoug01/Downloads/btcv')