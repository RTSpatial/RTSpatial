// Groute: An Asynchronous Multi-GPU Programming Framework
// http://www.github.com/groute/groute
// Copyright (c) 2017, A. Barak
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the names of the copyright holders nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef RTSPATIAL_UTILS_EVENT_POOL_H
#define RTSPATIAL_UTILS_EVENT_POOL_H
#include <cuda.h>

#include <future>
#include <utility>

#include "rtspatial/utils/exception.h"
#include "rtspatial/utils/stream.h"

namespace rtspatial {

struct IEvent {
  virtual ~IEvent() {}

  virtual void Wait(cudaStream_t stream) const = 0;
  virtual void Sync() const = 0;
  virtual bool Query() const = 0;
};

class EventHolder : public IEvent {
 private:
  const cudaEvent_t cuda_event;
  std::function<void(cudaEvent_t)> m_releaser;

 public:
  EventHolder(cudaEvent_t cuda_event,
              const std::function<void(cudaEvent_t)>& releaser)
      : cuda_event(cuda_event), m_releaser(releaser) {}

  ~EventHolder() { m_releaser(cuda_event); }

  void Wait(cudaStream_t stream) const override {
    CUDA_CHECK(cudaStreamWaitEvent(stream, cuda_event, 0));
  }

  void Sync() const override { CUDA_CHECK(cudaEventSynchronize(cuda_event)); }

  bool Query() const override {
    return cudaEventQuery(cuda_event) == cudaSuccess;
  }
};

class Event {
 private:
  std::shared_ptr<IEvent> m_internal_event;

 public:
  Event(cudaEvent_t cuda_event,
        const std::function<void(cudaEvent_t)>& releaser) {
    assert(cuda_event != nullptr);
    assert(releaser != nullptr);

    m_internal_event = std::make_shared<EventHolder>(cuda_event, releaser);
  }

  Event()
      : m_internal_event(nullptr)  // dummy event
  {}

  Event(const Event& other) : m_internal_event(other.m_internal_event) {}

  Event(Event&& other) : m_internal_event(std::move(other.m_internal_event)) {}

  Event& operator=(Event&& other) {
    this->m_internal_event = std::move(other.m_internal_event);
    return *this;
  }

  Event& operator=(const Event& other) {
    m_internal_event = other.m_internal_event;
    return *this;
  }

  Event(std::shared_ptr<IEvent> internal_event)
      : m_internal_event(std::move(internal_event)) {}

  static Event Record(cudaStream_t stream) {
    cudaEvent_t cuda_event;
    CUDA_CHECK(cudaEventCreateWithFlags(&cuda_event, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(cuda_event, stream));

    return Event(cuda_event,
                 [](cudaEvent_t ev)  // releaser, called on destruction event
                 { CUDA_CHECK(cudaEventDestroy(ev)); });
  }

  static Event Record(const Stream& stream) {
    return Record(stream.cuda_stream());
  }

  void Wait(cudaStream_t stream) const {
    if (m_internal_event == nullptr)
      return;  // dummy event
    m_internal_event->Wait(stream);
  }

  void Wait(const Stream& stream) const {
    if (m_internal_event == nullptr)
      return;  // dummy event
    m_internal_event->Wait(stream.cuda_stream());
  }

  void Sync() const {
    if (m_internal_event == nullptr)
      return;  // dummy event
    m_internal_event->Sync();
  }

  bool Query() const {
    if (m_internal_event == nullptr)
      return true;  // dummy event
    return m_internal_event->Query();
  }
};

/**
 * @brief Event-pool for managing and recycling per device cuda events
 */
class EventPool {
 private:
  std::vector<cudaEvent_t> m_events;
  std::deque<cudaEvent_t> m_pool;

  cudaEvent_t Create() const {
    cudaEvent_t ev;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    return ev;
  }

  void Destroy(cudaEvent_t ev) const { CUDA_CHECK(cudaEventDestroy(ev)); }

 public:
  EventPool(size_t cached_evs = 0) {
    m_events.reserve(cached_evs);
    CacheEvents(cached_evs);
  }

  void CacheEvents(size_t cached_evs) {
    for (size_t i = m_events.size(); i < cached_evs; i++) {
      m_events.push_back(Create());
      m_pool.push_back(m_events[i]);
    }
  }

  int GetCachedEventsNum() const { return (int) m_events.size(); }

  ~EventPool() {
    for (auto ev : m_events) {
      Destroy(ev);
    }
  }

  Event Record(cudaStream_t stream) {
    cudaEvent_t cuda_event;

    {  // guard block

      if (m_pool.empty()) {
        m_events.push_back(cuda_event = Create());
      } else {
        cuda_event = m_pool.front();
        m_pool.pop_front();
      }
    }

    // Record the event on the provided stream
    // The stream must be associated with the same device as the event
    CUDA_CHECK(cudaEventRecord(cuda_event, stream));

    return Event(cuda_event,
                 [this](cudaEvent_t ev)  // releaser, called on destruction of
                                         // internal EventHolder
                 { m_pool.push_back(ev); });
  }

  Event Record(const Stream& stream) { return Record(stream.cuda_stream()); }
};
}  // namespace rtspatial
#endif  // RTSPATIAL_UTILS_EVENT_POOL_H
