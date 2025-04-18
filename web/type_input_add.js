import { app } from "/scripts/app.js";

// 查找输入框或小部件的工具函数
const findInputByName = (node, name) => {
    return node.inputs ? node.inputs.find((w) => w.name === name) : null;
};

const findWidgetByName = (node, name) => {
    return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

// 控制输入框显示或隐藏的函数
function handleInputsVisibility(node, countValue, targets, type) {
    // 根据 countValue 动态更新输入端口
    for (let i = 1; i <= 50; i++) {
        targets.forEach((target) => {
            const name = `${target}_${i}`;
            const input = findInputByName(node, name);

            if (input) {
                if (i > countValue) {
                    node.removeInput(input);  // 正确地移除输入
                }
            } else {
                if (i <= countValue) {
                    node.addInput(name, type);  // 正确地添加输入
                }
            }
        });
    }
}

// 为新节点类型 `type_any_switch` 添加联动逻辑
app.registerExtension({
    name: "type",  // 注册的扩展名称

    nodeCreated(node) {


        if (node.constructor.title === "type_make_maskBatch") {
            if (node.widgets) {
                const countWidget = findWidgetByName(node, "count");
                let widgetValue = countWidget.value;
                handleInputsVisibility(node, widgetValue, ["mask"], "MASK");

                Object.defineProperty(countWidget, 'value', {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            handleInputsVisibility(node, newVal, ["mask"], "MASK");
                        }
                    }
                });
            }
        }



        if (node.constructor.title === "lay_image_match_W_and_H") {
            if (node.widgets) {
                const countWidget = findWidgetByName(node, "count");
                let widgetValue = countWidget.value;
                handleInputsVisibility(node, widgetValue, ["image"], "IMAGE");

                Object.defineProperty(countWidget, 'value', {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            handleInputsVisibility(node, newVal, ["image"], "IMAGE");
                        }
                    }
                });
            }
        }











        if (node.constructor.title === "AD_sch_prompt2") {
            if (node.widgets) {
                const countWidget = findWidgetByName(node, "count");
                let widgetValue = countWidget.value;
        
                // 根据 count 的值动态显示或隐藏 prompt 输入端口
                handleInputsVisibility(node, widgetValue, ["prompt"], "STRING");
        
                // 监听 count 的变化
                Object.defineProperty(countWidget, 'value', {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            handleInputsVisibility(node, newVal, ["prompt"], "STRING");
                        }
                    }
                });
            }
        }





        if (node.constructor.title === "type_condi_switch") {
            if (node.widgets) {
                const countWidget = findWidgetByName(node, "count");
                let widgetValue = countWidget.value;
                handleInputsVisibility(node, widgetValue, ["condition"], "CONDITION");

                Object.defineProperty(countWidget, 'value', {
                    get() {
                        return widgetValue;
                    },
                    set(newVal) {
                        if (newVal !== widgetValue) {
                            widgetValue = newVal;
                            handleInputsVisibility(node, newVal, ["condition"], "CONDITION");
                        }
                    }
                });
            }
        }




    },
});

