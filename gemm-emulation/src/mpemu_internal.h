#ifndef MPEMU_INTERNAL_H
#define MPEMU_INTERNAL_H

/* On Linux MPEMU_API is `visibility("default")`, which is correct both when
 * building the library and when consuming it. This guard only matters if the
 * project is ever built for Windows. */
#if defined(_WIN32)
#  undef MPEMU_API
#  define MPEMU_API __declspec(dllexport)
#endif

#endif /* MPEMU_INTERNAL_H */
