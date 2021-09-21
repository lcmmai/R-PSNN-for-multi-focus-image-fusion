from Net.NetUtils import *

Siamese_Branch_A = nn.Sequential(                                                                                       #TODO:Siamese_network_1
    conv7_block_1(ch_in=3, ch_out=16),
    conv_block_1(ch_in=16, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    RACP_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    RACP_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64)
)

Siamese_Branch_B = nn.Sequential(                                                                                       #TODO:Siamese_network_2
    conv7_block_1(ch_in=3, ch_out=16),
    conv_block_1(ch_in=16, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    RACP_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    RACP_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64)
)

Generation_Model = nn.Sequential(                                                                                       #TODO:Generation_network
    conv_block_1(ch_in=64+64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    res_block_1(ch_in=64, ch_out=64),
    conv1_block_1(ch_in=64, ch_out=64),
    conv7_block_1(ch_in=64, ch_out=3),
    conv_block_1(ch_in=3, ch_out=3),
)
