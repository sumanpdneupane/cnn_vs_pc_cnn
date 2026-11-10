import csv
import math

# (CSV column, history key, decimals). Same columns and rounding as the earlier 33 / 39 metrics CSVs.
COLUMNS = [('Epoch', 'epoch', None), ('Train Acc', 'train_acc', 4), ('Train Loss', 'train_loss', 4),
           ('Val Acc', 'val_acc', 2), ('Val Loss', 'val_loss', 4), ('Clean Train Acc', 'clean_train_acc', 3),
           ('Clean Train Loss', 'clean_train_loss', 4), ('LR', 'lr', 4)]

# Added only when the history has them (EP runs record time per epoch).
OPTIONAL = [('Train Time (s)', 'train_time_s', 1), ('Epoch Time (s)', 'epoch_time_s', 1)]


def _cell(value, decimals):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ''
    if decimals is None:
        return int(value)
    return round(float(value), decimals)


# Write the training history to a CSV, one row per epoch
def save_metrics_csv(history, path):
    columns = COLUMNS + [c for c in OPTIONAL if c[1] in history]
    best_acc, best_epoch = -1.0, 0
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _, _ in columns] + ['Best Val Acc', 'Best Epoch'])
        for i, epoch in enumerate(history['epoch']):
            if history['val_acc'][i] > best_acc:
                best_acc, best_epoch = history['val_acc'][i], epoch
            writer.writerow([_cell(history[key][i], d) for _, key, d in columns] + [round(best_acc, 2), int(best_epoch)])
    return path


# Read a metrics CSV back into a history dict
def load_metrics_csv(path):
    names = {name: (key, d) for name, key, d in COLUMNS + OPTIONAL}
    names.update({'Best Val Acc': ('best_val_acc', 2), 'Best Epoch': ('best_epoch', None)})
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    history = {}
    for column in rows[0].keys() if rows else []:
        key, d = names[column]
        history[key] = [None if r[column] == '' else (int(r[column]) if d is None else float(r[column])) for r in rows]
    history['generalization_gap'] = [None if c is None else c - v for c, v in zip(history['clean_train_acc'], history['val_acc'])]
    return history


# Print all epochs, then best val and mean val of the last epochs
def print_metrics(path, last_epochs=5):
    import pandas as pd
    history = load_metrics_csv(path)
    table = pd.DataFrame({name: history[key] for name, key, _ in COLUMNS + OPTIONAL if key in history})
    table['Gap (pp)'] = [None if g is None else round(g, 2) for g in history['generalization_gap']]
    table['Best Val Acc'], table['Best Epoch'] = history['best_val_acc'], history['best_epoch']
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 250):
        print(table.to_string(index=False, na_rep=''))
    val = history['val_acc'][-last_epochs:]
    first, last = history['epoch'][-len(val)], history['epoch'][-1]
    print(f"\nBest val: {history['best_val_acc'][-1]:.2f}% (epoch {history['best_epoch'][-1]}) | "
          f"Mean val epochs {first}-{last}: {sum(val) / len(val):.2f}% | Final val: {val[-1]:.2f}%")
