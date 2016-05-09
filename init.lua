require 'torch'

local kitti_scoring = {}

ks = require 'libkitti_scoring'
kitti_scoring.score_kitti = ks.score_kitti

function kitti_scoring.testme()
    print('testing')
end

return kitti_scoring
