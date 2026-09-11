// Test-only macOS pread interposer. Only the synthetic fixture is affected.
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <limits.h>
static atomic_int injected;
static ssize_t injected_pread(int fd, void *buf, size_t n, off_t offset) {
  char path[PATH_MAX];
  const char *mode = getenv("MLX_TEST_READ_FAULT");
  if (mode && fcntl(fd, F_GETPATH, path) == 0 && strstr(path, "/fault.safetensors")) {
    if (!atomic_exchange(&injected, 1)) {
      const char *message = "MLX_TEST_READ_FAULT injected\n";
      write(STDERR_FILENO, message, strlen(message));
      if (!strcmp(mode, "eintr")) { errno = EINTR; return -1; }
    }
    if (!strcmp(mode, "efault")) { errno = EFAULT; return -1; }
    if (!strcmp(mode, "short") && n > 4093) n = 4093;
  }
  return pread(fd, buf, n, offset);
}
__attribute__((used)) static struct { const void *replacement; const void *original; }
interpose __attribute__((section("__DATA,__interpose"))) = {
  (const void *)injected_pread, (const void *)pread
};
