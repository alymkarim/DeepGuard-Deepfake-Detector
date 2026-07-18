from src.split_dataset import split_paths
def test_no_overlap():
    p=[f'v{i}.mp4' for i in range(100)]; a,b,c=split_paths(p,42,.7,.15,.15); assert len(a)==70 and len(b)==15 and len(c)==15; assert set(a).isdisjoint(b) and set(a).isdisjoint(c) and set(b).isdisjoint(c)
