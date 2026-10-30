// Sites expects its source manifest and build output at the Git checkout root.
// The application itself remains in frontend/ for normal development.
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
const root = resolve(import.meta.dirname, '..');
const build = spawnSync('npm', ['run', 'build'], { cwd: resolve(root, 'frontend'), stdio:'inherit', env:process.env });
if(build.status !== 0) process.exit(build.status || 1);
rmSync(resolve(root,'dist'), {recursive:true,force:true});
cpSync(resolve(root,'frontend/dist'), resolve(root,'dist'), {recursive:true});
mkdirSync(resolve(root,'dist/.openai/drizzle'), {recursive:true});
cpSync(resolve(root,'frontend/drizzle'), resolve(root,'dist/.openai/drizzle'), {recursive:true});
