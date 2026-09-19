"""Reproduce both 640-case maps. Each group is persisted separately."""

from scan import scan

if __name__ == "__main__":
    for group in range(10):
        scan(group)
    for group in range(10):
        scan(group, dt=0.125, n=8192, tag="fine_")
