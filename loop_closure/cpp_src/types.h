#pragma once

#include <memory>

namespace radar_graph_slam {
    struct KeyFrame {
        int index;
    };
    // Definiamo l'alias per il puntatore qui, come tipo globale nel namespace
    using KeyFramePtr = std::shared_ptr<KeyFrame>;
}