import synapseclient
import synapseutils

syn = synapseclient.Synapse()
syn.login(authToken="")
files = synapseutils.syncFromSynapse(syn, 'syn18824258',path='/path/to/kcd_data/Downloads/orcascore/')