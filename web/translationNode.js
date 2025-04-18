// 导入ComfyUI的主应用对象，用于注册扩展和访问全局功能
import { app } from "/scripts/app.js";
// 导入ComfyWidgets，用于创建自定义UI小部件
import { ComfyWidgets } from "/scripts/widgets.js";

// 注册一个新的ComfyUI扩展
app.registerExtension({
    name: "ComfyUI-Apt_Collect", // 扩展的名称，用于标识此扩展

    // 在节点定义注册前执行的异步函数
    // nodeType: 节点类型对象
    // nodeData: 节点数据，包含节点的基本信息
    // app: ComfyUI的主应用对象
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 检查是否是翻译节点（中译英或英译中）
        if (nodeData.name === "ChineseToEnglish" || nodeData.name === "EnglishToChinese") {
            // 保存原始的onNodeCreated方法
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            // 重写onNodeCreated方法，在节点创建时进行初始化
            nodeType.prototype.onNodeCreated = function() {
                // 调用原始的onNodeCreated方法（如果存在）
                onNodeCreated?.apply(this, arguments);
                this.size = [300, 150]; // 设置节点的初始大小
                this.setDirtyCanvas(true, true); // 标记画布需要重绘
                this.updateInputOpacity(); // 初始化时更新输入框透明度
            };

            // 添加更新输入框透明度的方法
            nodeType.prototype.updateInputOpacity = function() {
                const inputWidget = this.widgets.find(w => w.name === "input_text");
                if (inputWidget && inputWidget.inputEl) {
                    const isOptionalConnected = this.inputs.find(input => input.name === "optional_input_text" && input.link !== null);
                    inputWidget.inputEl.style.opacity = isOptionalConnected ? "0.5" : "1";
                }
            };

            // 保存原始的onConnectionsChange方法
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            // 重写onConnectionsChange方法
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                // 调用原始的onConnectionsChange方法（如果存在）
                onConnectionsChange?.apply(this, arguments);
                this.updateInputOpacity(); // 连接变化时更新输入框透明度
            };

            // 定义用于更新翻译结果的函数
            function updateTranslation(translatedText) {
                // 确保translatedText是一个字符串
                if (Array.isArray(translatedText)) {
                    // 如果是数组，将其连接成字符串
                    translatedText = translatedText.join('');
                } else if (typeof translatedText !== 'string') {
                    // 如果不是字符串，强制转换为字符串
                    translatedText = String(translatedText);
                }

                // 如果结果小部件不存在，创建一个新的
                if (!this.resultWidget) {
                    // 使用ComfyWidgets创建一个新的STRING类型小部件
                    this.resultWidget = ComfyWidgets["STRING"](this, "translated_text", ["STRING", { multiline: true }], app).widget;
                    this.resultWidget.inputEl.readOnly = true; // 设置输入元素为只读
                    this.resultWidget.inputEl.style.opacity = 0.6; // 设置输入元素的不透明度
                }

                // 更新小部件的值为新的翻译结果
                this.resultWidget.value = translatedText;

                // 在下一帧中执行UI更新，以确保DOM已经准备好
                requestAnimationFrame(() => {
                    // 计算节点新的大小
                    const sz = this.computeSize();
                    // 确保新的大小不小于当前大小
                    sz[0] = Math.max(sz[0], this.size[0]);
                    sz[1] = Math.max(sz[1], this.size[1]);
                    // 调整节点大小
                    this.onResize?.(sz);
                    // 标记画布需要重绘
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            // 保存原始的onExecuted方法
            const onExecuted = nodeType.prototype.onExecuted;
            // 重写onExecuted方法，在节点执行完成后处理结果
            nodeType.prototype.onExecuted = function (message) {
                // 调用原始的onExecuted方法
                onExecuted?.apply(this, arguments);
                // 如果消息中包含文本，更新翻译结果
                if (message.text) {
                    updateTranslation.call(this, message.text);
                }
            };

            // 保存原始的onConfigure方法
            const onConfigure = nodeType.prototype.onConfigure;
            // 重写onConfigure方法，在节点配置加载时恢复状态
            nodeType.prototype.onConfigure = function (config) {
                // 调用原始的onConfigure方法
                onConfigure?.apply(this, arguments);
                // 如果配置中包含小部件值，使用它们更新显示
                if (config.widgets_values?.length > 1) {
                    updateTranslation.call(this, config.widgets_values[1]);
                }
                this.updateInputOpacity(); // 配置加载时更新输入框透明度
            };
        }
    },
});
