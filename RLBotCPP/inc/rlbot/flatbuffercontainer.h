#pragma once

#include "rlbot/interface.h"

#include <cstdlib>
#include <cstring>

namespace rlbot {
template <typename type> class FlatbufferContainer {
private:
  char *data;
  size_t size;
  const type *flatbuffer;

public:
  FlatbufferContainer(ByteBuffer buffer) {
    size = buffer.size;
    data = size > 0 ? (char *)malloc(size) : nullptr;
    if (size > 0)
      memcpy(data, buffer.ptr, size);

    flatbuffer = data ? flatbuffers::GetRoot<type>(data) : nullptr;
  }

  ~FlatbufferContainer() { free(data); }

  FlatbufferContainer(const FlatbufferContainer &flatbuffercontainer) {
    size = flatbuffercontainer.size;
    data = size > 0 ? (char *)malloc(size) : nullptr;
    if (size > 0)
      memcpy(data, flatbuffercontainer.data, size);

    flatbuffer = data ? flatbuffers::GetRoot<type>(data) : nullptr;
  }

  FlatbufferContainer(FlatbufferContainer &&flatbuffercontainer) {
    size = flatbuffercontainer.size;
    data = flatbuffercontainer.data;

    flatbuffer = data ? flatbuffers::GetRoot<type>(data) : nullptr;

    flatbuffercontainer.data = nullptr;
    flatbuffercontainer.size = 0;
    flatbuffercontainer.flatbuffer = nullptr;
  }

  FlatbufferContainer<type> &
  operator=(const FlatbufferContainer &flatbuffercontainer) {
    if (this == &flatbuffercontainer)
      return *this;

    free(data);

    size = flatbuffercontainer.size;
    data = size > 0 ? (char *)malloc(size) : nullptr;
    if (size > 0)
      memcpy(data, flatbuffercontainer.data, size);

    flatbuffer = data ? flatbuffers::GetRoot<type>(data) : nullptr;

    return *this;
  }

  FlatbufferContainer<type> &
  operator=(FlatbufferContainer &&flatbuffercontainer) {
    if (this == &flatbuffercontainer)
      return *this;

    free(data);

    size = flatbuffercontainer.size;
    data = flatbuffercontainer.data;

    flatbuffer = data ? flatbuffers::GetRoot<type>(data) : nullptr;

    flatbuffercontainer.data = nullptr;
    flatbuffercontainer.size = 0;
    flatbuffercontainer.flatbuffer = nullptr;

    return *this;
  }

  const type *getRoot() const { return flatbuffer; }
  const type *operator->() const { return flatbuffer; }
};
} // namespace rlbot
