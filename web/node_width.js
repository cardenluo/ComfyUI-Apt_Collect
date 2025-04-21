import { app } from '../../scripts/app.js';

app.registerExtension({
    name: 'apt.node_width',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 检查是否是Apt_Preset节点
        if (nodeData.name && (
            nodeData.name.startsWith('sum_') || 
            nodeData.name.startsWith('Data_') || 
            nodeData.name.startsWith('AD_') || 
            nodeData.name.startsWith('basic_') || 
            nodeData.name.startsWith('chx_') || 
            nodeData.name.startsWith('Apply_') || 
            nodeData.name.startsWith('Stack_') || 
            nodeData.name.startsWith('param_') || 
            nodeData.name.startsWith('Model_') || 
            nodeData.name.startsWith('CN_') || 
            nodeData.name.startsWith('photoshop_') || 
            nodeData.name.startsWith('load_')
        )) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const r = onNodeCreated?.apply(this, arguments);
                
                // 设置固定宽度
                this.size[0] = 300;
                this.setSize([300, this.size[1]]);
                
                // 禁用节点的宽度调整
                const originalComputeSize = this.computeSize;
                this.computeSize = function() {
                    const size = originalComputeSize.call(this);
                    size[0] = 300; // 保持宽度固定
                    return size;
                };
                
                return r;
            };
        }
    }
});