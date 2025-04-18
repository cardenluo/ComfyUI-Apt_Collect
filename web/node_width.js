import { app } from '../../scripts/app.js';

app.registerExtension({
    name: 'apt.node_width',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 检查是否是Apt_Preset节点
        if (nodeData.name && (
            nodeData.name.startsWith('IO_') ||
            nodeData.name.startsWith('view_') ||
            nodeData.name.startsWith('pack_') ||
            nodeData.name.startsWith('list_') ||
            nodeData.name.startsWith('batch_') ||
            nodeData.name.startsWith('type_') ||
            nodeData.name.startsWith('math_') ||
            nodeData.name.startsWith('model_') ||
            nodeData.name.startsWith('Image_') ||
            nodeData.name.startsWith('Mask_') ||
            nodeData.name.startsWith('latent_') ||
            nodeData.name.startsWith('text_') ||
            nodeData.name.startsWith('stack_') ||
            nodeData.name.startsWith('color_') ||
            nodeData.name.startsWith('img_') ||
            nodeData.name.startsWith('lay_') ||
            nodeData.name.startsWith('pad_') ||
            nodeData.name.startsWith('sampler_') ||
            nodeData.name.startsWith('AD_')
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