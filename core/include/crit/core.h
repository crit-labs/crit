#pragma once
// Core's native interface, in parts by concern (each one self-contained): the resource ledger and backend services,
// the engine profile, request options and scoring queries, the Model/Session interface, the ConcurrentModel
// interface, and the Engine with its handles, results and lifecycle trace.
#include "crit/core/resources.h"
#include "crit/core/profile.h"
#include "crit/core/request.h"
#include "crit/core/model.h"
#include "crit/core/concurrent.h"
#include "crit/core/engine.h"
