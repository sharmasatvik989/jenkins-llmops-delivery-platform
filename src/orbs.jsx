import React from "react";
import { createRoot } from "react-dom/client";
import { ThinkingOrb } from "thinking-orbs";

const states = ["searching", "connecting", "shaping", "weaving", "working", "solving"];
const labels = [
  "Discovering compatible models",
  "Connecting model metadata",
  "Shaping infrastructure requirements",
  "Weaving release configuration",
  "Validating the Jenkins release",
  "Solving Kubernetes delivery",
];

document.querySelectorAll(".thought-orb").forEach((container, index) => {
  container.replaceChildren();
  createRoot(container).render(
    <ThinkingOrb
      state={states[index]}
      size={64}
      theme="dark"
      aria-label={labels[index]}
    />,
  );
});
