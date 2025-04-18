import { app } from "../../scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => {
  return node.widgets ? node.widgets.find((w) => w.name === name) : null;
};

const doesInputWithNameExist = (node, name) => {
  return node.inputs ? node.inputs.some((input) => input.name === name) : false;
};

const HIDDEN_TAG = "tschide";

function toggleWidget(node, widget, show = false, suffix = "") {
  if (!widget || doesInputWithNameExist(node, widget.name)) return;

  if (!origProps[widget.name]) {
    origProps[widget.name] = {
      origType: widget.type,
      origComputeSize: widget.computeSize,
    };
  }

  widget.type = show ? origProps[widget.name].origType : HIDDEN_TAG + suffix;
  widget.computeSize = show
    ? origProps[widget.name].origComputeSize
    : () => [0, -4];
  widget.linkedWidgets?.forEach((w) =>
    toggleWidget(node, w, ":" + widget.name, show)
  );

  const newHeight = node.computeSize()[1];
  node.setSize([node.size[0], newHeight]);
}

function handleVisibility(node, countValue, node_type) {
  const baseNamesMap = {
    stack_text_combine: ["text"],
    stack_Wildcards: ["wildcard_name"],
    Stack_LoRA2: ["lora"],
  };

  const baseNames = baseNamesMap[node_type];

  if (node_type === "stack_text_combine") {
    for (let i = 1; i <= 50; i++) {
      const textWidget = findWidgetByName(node, `${baseNames[0]}_${i}`);
      if (textWidget) {
        toggleWidget(node, textWidget, i <= countValue);
      }
    }
  } else if (node_type === "stack_Wildcards") {
    for (let i = 1; i <= 50; i++) {
      const wildcardWidget = findWidgetByName(node, `${baseNames[0]}_${i}`);
      if (wildcardWidget) {
        toggleWidget(node, wildcardWidget, i <= countValue);
      }
    }
  } else if (node_type === "Stack_LoRA2") {
    for (let i = 1; i <= 10; i++) {
      const nameWidget = findWidgetByName(node, `${baseNames[0]}_${i}_name`);
      const strengthWidget = findWidgetByName(node, `${baseNames[0]}_${i}_strength`);
      if (nameWidget && strengthWidget) {
        toggleWidget(node, nameWidget, i <= countValue);
        toggleWidget(node, strengthWidget, i <= countValue);
      }
    }
  }
}

const nodeWidgetHandlers = {
  stack_text_combine: {
    text_count: (node, widget) => handleVisibility(node, widget.value, "stack_text_combine"),
  },
  stack_Wildcards: {
    wildcards_count: (node, widget) => handleVisibility(node, widget.value, "stack_Wildcards"),
  },
  Stack_LoRA2: {
    num_loras: (node, widget) => handleVisibility(node, widget.value, "Stack_LoRA2"),
  },
};

function widgetLogic(node, widget) {
  const handler = nodeWidgetHandlers[node.comfyClass]?.[widget.name];
  if (handler) {
    handler(node, widget);
  }
}

app.registerExtension({
  name: "stack.widgethider",
  nodeCreated(node) {
    for (const w of node.widgets || []) {
      let widgetValue = w.value;
      let originalDescriptor = Object.getOwnPropertyDescriptor(w, "value");
      if (!originalDescriptor) {
        originalDescriptor = Object.getOwnPropertyDescriptor(
          w.constructor.prototype,
          "value"
        );
      }
      widgetLogic(node, w);
      Object.defineProperty(w, "value", {
        get() {
          let valueToReturn =
            originalDescriptor && originalDescriptor.get
              ? originalDescriptor.get.call(w)
              : widgetValue;
          return valueToReturn;
        },
        set(newVal) {
          if (originalDescriptor && originalDescriptor.set) {
            originalDescriptor.set.call(w, newVal);
          } else {
            widgetValue = newVal;
          }
          widgetLogic(node, w);
        },
      });
    }
    setTimeout(() => {
      initialized = true;
    }, 500);
  },
});