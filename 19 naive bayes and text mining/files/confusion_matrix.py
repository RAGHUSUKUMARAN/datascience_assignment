import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion matrix — Naive Bayes")
plt.colorbar()
plt.xticks(range(len(classes)), classes, rotation=90)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("nb_confusion_matrix.png")
plt.close()


# baseline_metrics and tuned_metrics are dicts like the 'summary' above
comp_df = pd.DataFrame([
    {"model":"baseline", **baseline_metrics},
    {"model":"tuned", **tuned_metrics}
])
comp_df.to_csv("baseline_vs_tuned_metrics.csv", index=False)
