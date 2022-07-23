from tokenize import group


def get_flatten_classes(config):
    groups = config.Class.groups
    classes = config.Class.classes
    assert set(classes.keys()) == set(groups), (classes, groups)
    assert len(groups) == len(set(groups)), groups
    assert len(classes.keys()) == len(set(classes.keys())), classes

    classes_flatten = []
    for g in groups:
        classes_flatten.extend(classes[g])
    assert len(classes_flatten) == len(set(classes_flatten)), classes_flatten
    return classes_flatten
