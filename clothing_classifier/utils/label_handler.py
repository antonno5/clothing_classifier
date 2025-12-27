def get_labels_mapping() -> dict[int, str]:
    return {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }


def id_to_label(id: int) -> str:
    return get_labels_mapping()[id]
