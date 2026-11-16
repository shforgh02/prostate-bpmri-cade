# Resumable transfers to and from a Modal volume. Only missing files are sent or fetched; every chunk is retried,
# so an internet drop or a restart loses nothing. The workspace is chosen by the MODAL_PROFILE environment variable.
#   python analysis/_modal_sync.py count <volume> <remote_dir> <suffix>
#   python analysis/_modal_sync.py up    <volume> <local_dir> <remote_dir>            (all *.nii.gz in local_dir)
#   python analysis/_modal_sync.py down  <volume> <remote_dir> <local_dir> <suffix1,suffix2>
import sys, os, time
import modal

cmd, volname = sys.argv[1], sys.argv[2]
vol = modal.Volume.from_name(volname, create_if_missing=True)


def remote(rdir):
    """name -> size for the files directly inside rdir ({} if the folder does not exist yet)."""
    try:
        out = {}
        for e in vol.listdir(rdir):
            out[e.path.replace('\\', '/').split('/')[-1]] = getattr(e, 'size', None)
        return out
    except Exception as ex:
        if 'no such' in str(ex).lower() or 'not found' in str(ex).lower():
            return {}
        raise


def retry(fn, what, tries=40, wait=60):
    for k in range(tries):
        try:
            return fn()
        except Exception as ex:
            print(f'  retry {k + 1}/{tries} after error in {what}: {str(ex)[:150]}', flush=True)
            time.sleep(wait)
    raise RuntimeError('gave up: ' + what)


if cmd == 'put':                                    # python analysis/_modal_sync.py put <volume> <local_file> <remote_path>
    def one():
        with vol.batch_upload(force=True) as b:
            b.put_file(sys.argv[3], '/' + sys.argv[4].strip('/'))
    retry(one, 'put')
    print('put done')
elif cmd == 'count':
    r = retry(lambda: remote(sys.argv[3]), 'list')
    print(sum(1 for n in r if n.endswith(sys.argv[4])))
elif cmd == 'up':
    local, rdir = sys.argv[3], sys.argv[4].strip('/')
    have = retry(lambda: remote(rdir), 'list')
    files = sorted(f for f in os.listdir(local) if f.endswith('.nii.gz'))
    miss = [f for f in files if f not in have or (have[f] is not None and have[f] != os.path.getsize(os.path.join(local, f)))]
    print(f'up {volname}:{rdir} | local {len(files)} | already there {len(files) - len(miss)} | to send {len(miss)}', flush=True)
    for i in range(0, len(miss), 30):
        chunk = miss[i:i + 30]

        def send():
            with vol.batch_upload(force=True) as b:
                for f in chunk:
                    b.put_file(os.path.join(local, f), f'/{rdir}/{f}')
        retry(send, 'upload')
        print(f'  sent {min(i + 30, len(miss))}/{len(miss)}', flush=True)
    print('up done')
elif cmd == 'down':
    rdir, local, suf = sys.argv[3].strip('/'), sys.argv[4], tuple(sys.argv[5].split(','))
    os.makedirs(local, exist_ok=True)
    r = retry(lambda: remote(rdir), 'list')
    want = sorted(n for n in r if n.endswith(suf))
    miss = [n for n in want if not os.path.exists(os.path.join(local, n)) or (r[n] is not None and os.path.getsize(os.path.join(local, n)) != r[n])]
    print(f'down {volname}:{rdir} | remote {len(want)} | to fetch {len(miss)}', flush=True)
    for k, n in enumerate(miss):
        def fetch():
            tmp = os.path.join(local, n + '.part')
            with open(tmp, 'wb') as f:
                for blk in vol.read_file(f'{rdir}/{n}'):
                    f.write(blk)
            os.replace(tmp, os.path.join(local, n))
        retry(fetch, 'download ' + n)
        if (k + 1) % 100 == 0:
            print(f'  fetched {k + 1}/{len(miss)}', flush=True)
    print('down done')
